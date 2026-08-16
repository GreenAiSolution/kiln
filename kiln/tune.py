"""
kiln.tune - the autotuner, and the cost model it learns.

Choosing a schedule is a search problem: unroll factor, prefetch distance,
whether reductions carry a compensation term. Fifty-odd candidates per
kernel, and the right answer moves with the size of the data, the shape of
the expression and the machine underneath.

Measuring all of them is correct and slow. So KILN measures some of them
once, fits a model that predicts runtime from features of the *generated
code* - instructions per element, loads per element, register pressure,
working-set size - and then uses that model to shortlist candidates for
kernels it has never run. Only the shortlist gets timed.

The model is ridge regression on log(time per element), solved in closed
form. It is deliberately simple: with a few hundred samples anything with
more capacity memorises. What matters is that it is fitted on measurements
from *this* machine, and that bench/autotune_study.py scores it by
leave-one-kernel-out cross-validation - trained without ever seeing the
kernel it is then asked to schedule.

Pure standard library.
"""

import json
import math
import os
import time

from .lower import Schedule
from .runtime import bench_batched, compile as kcompile

MODEL_PATH = os.path.expanduser("~/.kiln/cost_model.json")

UNROLLS = (1, 2, 3, 4, 5, 6, 8, 10)
PREFETCHES = (0, 128, 256, 512, 1024)


def candidates(is_reduce=False):
    out = []
    for u in UNROLLS:
        for pf in PREFETCHES:
            out.append(Schedule(u, pf))
    return out


# The model's job is to rank schedules *within* one kernel at one size, so
# every feature that matters has to describe how a schedule knob interacts
# with a property of the kernel. A feature that is constant across the
# candidates for a given kernel carries no ranking information at all.
FEATURES = [
    "bias",
    "unroll", "inv_unroll", "log_unroll",
    "prefetch_on", "log_prefetch",
    "insns_per_elem", "ld_per_elem", "st_per_elem", "alu_per_elem",
    "reg_pressure", "spare_regs",
    "log_blocks", "leftover_frac", "tail_python",
    # schedule knob x kernel property
    "unroll_x_insns", "unroll_x_ld", "unroll_x_intensity",
    "unroll_x_logn", "unroll_x_logws", "unroll_x_exp", "unroll_x_reduce",
    "invunroll_x_insns", "invunroll_x_logn",
    "logunroll_x_logblocks",
    "pf_x_logws", "pf_x_ld", "pf_x_intensity", "pf_x_logn",
    "regpress_sq",
]


def featurise(meta, sched, n, nbufs, is_reduce):
    ipe = meta["insns_per_elem"]
    lpe = meta["ld_per_elem"]
    spe = meta["st_per_elem"]
    ape = meta["alu_per_elem"]
    U = float(meta["unroll"])
    ws = max(1.0, n * 4.0 * max(1, nbufs))
    logn = math.log2(max(n, 2))
    logws = math.log2(ws)
    nvec = max(1, n // 4)
    blocks = max(1, meta["blocks"])
    logb = math.log2(blocks)
    intensity = ape / max(lpe + spe, 0.25)
    regp = meta["live"] * U / 32.0
    pf = math.log2(1.0 + sched.prefetch / 64.0)
    pon = 1.0 if sched.prefetch else 0.0
    f = {
        "bias": 1.0,
        "unroll": U,
        "inv_unroll": 1.0 / U,
        "log_unroll": math.log2(U),
        "prefetch_on": pon,
        "log_prefetch": pf,
        "insns_per_elem": ipe,
        "ld_per_elem": lpe,
        "st_per_elem": spe,
        "alu_per_elem": ape,
        "reg_pressure": regp,
        "spare_regs": (32.0 - meta["regs_used"]) / 32.0,
        "log_blocks": logb,
        "leftover_frac": meta["leftover"] / nvec,
        "tail_python": 1.0 if meta["tail_python"] else 0.0,
        "unroll_x_insns": U * ipe,
        "unroll_x_ld": U * lpe,
        "unroll_x_intensity": U * intensity,
        "unroll_x_logn": U * logn,
        "unroll_x_logws": U * logws,
        "unroll_x_exp": U * (1.0 if meta["has_exp"] else 0.0),
        "unroll_x_reduce": U * (1.0 if is_reduce else 0.0),
        "invunroll_x_insns": ipe / U,
        "invunroll_x_logn": logn / U,
        "logunroll_x_logblocks": math.log2(U) * logb,
        "pf_x_logws": pf * logws,
        "pf_x_ld": pf * lpe,
        "pf_x_intensity": pf * intensity,
        "pf_x_logn": pf * logn,
        "regpress_sq": regp * regp,
    }
    return [f[k] for k in FEATURES]


# ------------------------------------------------------------ linear algebra

def solve(A, b, ridge=1e-3):
    """Ridge-regularised normal equations, Gaussian elimination with
    partial pivoting. The ridge term is what keeps this stable when two
    features are nearly collinear, which they are."""
    m = len(A[0])
    AtA = [[0.0] * m for _ in range(m)]
    Atb = [0.0] * m
    for row, y in zip(A, b):
        for i in range(m):
            ri = row[i]
            if ri:
                Atb[i] += ri * y
                Ai = AtA[i]
                for j in range(m):
                    Ai[j] += ri * row[j]
    for i in range(m):
        AtA[i][i] += ridge
    M = [AtA[i] + [Atb[i]] for i in range(m)]
    for c in range(m):
        p = max(range(c, m), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-14:
            M[c][c] += 1.0
            p = c
        M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        for r in range(m):
            if r == c:
                continue
            f = M[r][c] / piv
            if f:
                for k in range(c, m + 1):
                    M[r][k] -= f * M[c][k]
    return [M[i][m] / M[i][i] for i in range(m)]


# ------------------------------------------------------------------ model

class CostModel:
    def __init__(self, w=None, n_samples=0, machine=""):
        self.w = w or [0.0] * len(FEATURES)
        self.n_samples = n_samples
        self.machine = machine
        self.fitted = w is not None

    def predict(self, feats):
        """Predicted log of nanoseconds per element."""
        return sum(wi * fi for wi, fi in zip(self.w, feats))

    def fit(self, X, y, ridge=1e-3):
        self.w = solve(X, y, ridge)
        self.n_samples = len(X)
        self.fitted = True
        return self

    def r2(self, X, y):
        pred = [self.predict(x) for x in X]
        mu = sum(y) / len(y)
        ss_res = sum((a - b) ** 2 for a, b in zip(y, pred))
        ss_tot = sum((v - mu) ** 2 for v in y)
        return 1.0 - ss_res / ss_tot if ss_tot else 0.0

    def save(self, path=MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"features": FEATURES, "w": self.w,
                       "n_samples": self.n_samples,
                       "machine": self.machine,
                       "saved": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=1)
        return path

    @classmethod
    def load(cls, path=MODEL_PATH):
        if not os.path.exists(path):
            return cls()
        with open(path) as f:
            d = json.load(f)
        if d.get("features") != FEATURES:
            return cls()          # feature set changed; the old fit is void
        return cls(d["w"], d.get("n_samples", 0), d.get("machine", ""))

    def weights(self):
        return dict(zip(FEATURES, self.w))


# ------------------------------------------------------------------ tuning

def compile_candidate(prog, sched):
    try:
        return kcompile(prog, sched, contract=False)
    except Exception:
        return None


def _shape(prog):
    n = prog.stages[0].n
    nbufs = len(prog.buffers())
    is_reduce = any(s.kind == "reduce" for s in prog.stages)
    return n, nbufs, is_reduce


def measure(prog, sched, bufs, seconds=0.03):
    c = compile_candidate(prog, sched)
    if c is None:
        return None, None
    c.bind(bufs)
    r = bench_batched(lambda: c.run(), seconds=seconds)
    return c, r["best"]


def exhaustive(prog, bufs, seconds=0.03, verbose=False):
    """Measure every candidate. The ground truth the model is scored against."""
    n, nbufs, is_reduce = _shape(prog)
    results = []
    for sched in candidates(is_reduce):
        c, t = measure(prog, sched, bufs, seconds)
        if c is None:
            continue
        feats = featurise(c.kernels[0].meta, sched, n, nbufs, is_reduce)
        results.append({"sched": sched, "time": t, "feats": feats,
                        "meta": c.kernels[0].meta})
        if verbose:
            print(f"    {sched} -> {t * 1e6:.2f} us")
    results.sort(key=lambda r: r["time"])
    return results


def guided(prog, bufs, model, shortlist=3, seconds=0.03):
    """Let the model rank every candidate, then measure only the top few.

    This is the point of the model: compiling a candidate is cheap (about
    half a millisecond), measuring one is not. Ranking is free.
    """
    n, nbufs, is_reduce = _shape(prog)
    scored = []
    t0 = time.perf_counter()
    for sched in candidates(is_reduce):
        c = compile_candidate(prog, sched)
        if c is None:
            continue
        feats = featurise(c.kernels[0].meta, sched, n, nbufs, is_reduce)
        scored.append((model.predict(feats), sched, c))
    scored.sort(key=lambda s: s[0])
    rank_s = time.perf_counter() - t0

    best = None
    for _, sched, c in scored[:shortlist]:
        c.bind(bufs)
        t = bench_batched(lambda: c.run(), seconds=seconds)["best"]
        if best is None or t < best[1]:
            best = (sched, t, c)
    return {"sched": best[0], "time": best[1], "compiled": best[2],
            "ranked": len(scored), "measured": min(shortlist, len(scored)),
            "rank_seconds": rank_s,
            "top_choice": scored[0][1] if scored else None}


def autotune(prog, bufs, model=None, shortlist=3, seconds=0.03):
    """The normal entry point: use the saved model if there is one, fall back
    to measuring everything if there is not."""
    model = model or CostModel.load()
    if model.fitted:
        return guided(prog, bufs, model, shortlist, seconds)
    res = exhaustive(prog, bufs, seconds)
    return {"sched": res[0]["sched"], "time": res[0]["time"],
            "compiled": None, "ranked": len(res), "measured": len(res),
            "rank_seconds": 0.0, "top_choice": res[0]["sched"]}
