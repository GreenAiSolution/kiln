"""
Does the cost model actually know anything?

The easy way to make an autotuner look good is to train it on the kernel you
then ask it to schedule. That measures memorisation, not knowledge. So this
study holds a whole kernel family out: the model is fitted on measurements
from every *other* kernel, and then asked to pick a schedule for the one it
has never seen, at a size it has never seen.

Three numbers per held-out kernel:

  default    the schedule you get with no tuning at all
  guided     what the model picks after ranking (compiling, not timing) all
             40 candidates and measuring only its top 3
  best       the fastest of all 40, found by measuring all of them

"guided/best" is the score that matters. 1.00 means the model found the
optimum. The cost of guided is also reported, because an autotuner that
takes longer than the speedup it finds is not an autotuner.
"""

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kiln import ir, jit                                       # noqa: E402
from kiln.lower import Schedule                                # noqa: E402
from kiln.tune import (CostModel, FEATURES, exhaustive,        # noqa: E402
                       featurise, guided, candidates)
from kiln.runtime import compile as kcompile                   # noqa: E402


# ------------------------------------------------------------- the kernels

def k_axpy(p, n):
    p.map([("out", ir.load("a") * 2.5 + ir.load("b"))], n)


def k_chain(p, n):
    a, b, c = ir.load("a"), ir.load("b"), ir.load("c")
    p.map([("out", (a * b + c) * a - b)], n)


def k_wide(p, n):
    a, b, c, d = (ir.load(x) for x in "abcd")
    p.map([("out", (a * b + c * d) * (a - d) + (b * c - a * a))], n)


def k_poly(p, n):
    x = ir.load("a")
    p.map([("out", ((((x * 0.5 + 1.5) * x - 2.25) * x + 3.125) * x - 0.75))], n)


def k_sigmoid(p, n):
    p.map([("out", ir.sigmoid(ir.load("a")))], n)


def k_gelu(p, n):
    p.map([("out", ir.gelu(ir.load("a")))], n)


def k_hypot(p, n):
    a, b = ir.load("a"), ir.load("b")
    p.map([("out", ir.sqrt(a * a + b * b))], n)


def k_sum(p, n):
    p.reduce("s", "sum", ir.load("a"), n)


def k_sumsq(p, n):
    a, b = ir.load("a"), ir.load("b")
    d = a - b
    p.reduce("s", "sum", d * d, n)


def k_dot(p, n):
    p.reduce("s", "sum", ir.load("a") * ir.load("b"), n)


def k_multi(p, n):
    a, b = ir.load("a"), ir.load("b")
    s = a + b
    p.map([("o1", s * s), ("o2", s - a * b), ("o3", ir.relu(s))], n)


def k_scale_relu(p, n):
    a, b = ir.load("a"), ir.load("b")
    p.map([("out", ir.relu(a * 3.0 - b) * 0.5 + ir.relu(b))], n)


KERNELS = [
    ("axpy", k_axpy), ("chain", k_chain), ("wide", k_wide), ("poly", k_poly),
    ("sigmoid", k_sigmoid), ("gelu", k_gelu), ("hypot", k_hypot),
    ("sum", k_sum), ("sumsq", k_sumsq), ("dot", k_dot),
    ("multi", k_multi), ("scale_relu", k_scale_relu),
]

SIZES = [4096, 65536, 1 << 19, 1 << 22]


def make_bufs(p, n):
    bufs = {}
    for i, nm in enumerate(p.buffers()):
        b = jit.Buf(n)
        for j in range(0, n, 97):
            b[j] = 0.5 + (i + j % 13) * 0.01
        bufs[nm] = b
    return bufs


def collect(seconds):
    """Measure every (kernel, size, schedule). This is the slow part."""
    samples = []
    for name, fn in KERNELS:
        for n in SIZES:
            p = ir.Program(name)
            fn(p, n)
            ir.contract_program(p)
            bufs = make_bufs(p, n)
            res = exhaustive(p, bufs, seconds=seconds)
            for r in res:
                samples.append({
                    "kernel": name, "n": n, "sched": r["sched"].key(),
                    "time": r["time"], "feats": r["feats"],
                })
            print(f"  {name:<11} n={n:<9} "
                  f"best {res[0]['time'] * 1e6:8.2f}us with {res[0]['sched']}  "
                  f"worst {res[-1]['time'] * 1e6:8.2f}us  "
                  f"spread {res[-1]['time'] / res[0]['time']:.2f}x")
    return samples


def target(s, group_mean):
    """log-time, with the per-(kernel,size) mean removed.

    Absolute runtime is dominated by which kernel and which size this is -
    facts the model cannot change. What it has to predict is the *relative*
    effect of a schedule, so the group mean is subtracted and the model is
    left with exactly the signal it is being asked for.
    """
    return math.log(s["time"]) - group_mean[(s["kernel"], s["n"])]


def group_means(samples):
    acc = {}
    for s in samples:
        acc.setdefault((s["kernel"], s["n"]), []).append(math.log(s["time"]))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=0.02)
    ap.add_argument("--shortlist", type=int, default=3)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    print("Collecting measurements (every kernel x size x schedule)")
    print("=" * 84)
    t0 = time.perf_counter()
    samples = collect(args.seconds)
    collect_s = time.perf_counter() - t0
    print(f"\n{len(samples)} measurements in {collect_s:.1f}s")

    # how much is even at stake: best vs worst schedule, per kernel and size
    spreads = {}
    for s in samples:
        k = (s["kernel"], s["n"])
        lo, hi = spreads.get(k, (1e9, 0.0))
        spreads[k] = (min(lo, s["time"]), max(hi, s["time"]))
    sp = sorted(hi / lo for lo, hi in spreads.values())
    print(f"schedule spread (worst/best): median {sp[len(sp) // 2]:.2f}x   "
          f"max {sp[-1]:.2f}x   min {sp[0]:.2f}x")
    print("   ^ this is the entire prize. A tuner cannot beat the best "
          "schedule, only find it.")

    gm = group_means(samples)
    X = [s["feats"] for s in samples]
    y = [target(s, gm) for s in samples]
    full = CostModel().fit(X, y)
    print(f"in-sample R^2 : {full.r2(X, y):.4f}")

    # ---- leave one kernel out
    print("\nLeave-one-kernel-out: the model never sees the kernel it schedules")
    print("=" * 84)
    print(f"  {'held out':<12} {'default us':>11} {'guided us':>10} "
          f"{'best us':>9} {'guided/best':>12} {'vs default':>11} {'R^2':>7}")
    rows = []
    for name, fn in KERNELS:
        tr = [s for s in samples if s["kernel"] != name]
        te = [s for s in samples if s["kernel"] == name]
        m = CostModel().fit([s["feats"] for s in tr],
                            [target(s, gm) for s in tr])
        r2 = m.r2([s["feats"] for s in te], [target(s, gm) for s in te])

        for n in SIZES:
            p = ir.Program(name)
            fn(p, n)
            ir.contract_program(p)
            bufs = make_bufs(p, n)

            best_t = min(s["time"] for s in te if s["n"] == n)
            g = guided(p, bufs, m, shortlist=args.shortlist,
                       seconds=args.seconds)
            c = kcompile(p, Schedule(4), contract=False)
            c.bind(bufs)
            from kiln.runtime import bench_batched
            def_t = bench_batched(lambda: c.run(), seconds=args.seconds)["best"]

            rows.append({
                "kernel": name, "n": n, "default": def_t, "guided": g["time"],
                "best": best_t, "ratio": g["time"] / best_t,
                "vs_default": def_t / g["time"], "r2": r2,
                "ranked": g["ranked"], "measured": g["measured"],
                "rank_seconds": g["rank_seconds"],
                "guided_sched": str(g["sched"]),
            })
            print(f"  {name:<12} {def_t * 1e6:>11.2f} {g['time'] * 1e6:>10.2f} "
                  f"{best_t * 1e6:>9.2f} {g['time'] / best_t:>11.3f}x "
                  f"{def_t / g['time']:>10.2f}x {r2:>7.3f}")

    ratios = [r["ratio"] for r in rows]
    vd = [r["vs_default"] for r in rows]
    ratios.sort()
    print("\n" + "=" * 84)
    print(f"cases                        : {len(rows)}")
    print(f"guided / exhaustive-best     : median {ratios[len(ratios) // 2]:.3f}x"
          f"   mean {sum(ratios) / len(ratios):.3f}x   worst {max(ratios):.3f}x")
    hit = sum(1 for r in ratios if r <= 1.02)
    print(f"within 2% of the true optimum: {hit}/{len(rows)} "
          f"({100 * hit / len(rows):.0f}%)")
    print(f"speedup over the default     : median {sorted(vd)[len(vd) // 2]:.2f}x"
          f"   best {max(vd):.2f}x")
    print(f"candidates ranked per kernel : {rows[0]['ranked']}, "
          f"of which {rows[0]['measured']} were actually timed")
    print(f"ranking cost                 : "
          f"{sum(r['rank_seconds'] for r in rows) / len(rows) * 1e3:.0f} ms "
          f"(compiles all candidates, times none)")

    print("\nWhat the model learned (largest weights on log ns/element):")
    w = full.weights()
    for k in sorted(w, key=lambda k: -abs(w[k]))[:8]:
        print(f"  {k:<18} {w[k]:+.4f}")

    full.machine = "Apple M1 Max"
    path = full.save()
    print(f"\nmodel saved to {path} ({full.n_samples} samples)")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"rows": rows, "weights": w,
                       "in_sample_r2": full.r2(X, y),
                       "n_samples": len(samples)}, f, indent=1)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
