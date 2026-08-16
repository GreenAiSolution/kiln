"""
kiln.lower - IR to machine code.

For each stage KILN builds one loop that does all the work in registers.
The passes, in order:

  1. contract    add(mul(a,b),c) -> fma
  2. liveness    when each value is born and when it dies
  3. unroll      how many 4-wide vectors to run in flight at once
  4. allocate    linear scan over the 32 NEON registers
  5. specialise  the loop trip count is a compile-time constant, because a
                 JIT knows the shape it was called with
  6. emit        node-major, so the unrolled copies interleave and the
                 out-of-order core has independent work every cycle

Register budget: constants and loop-carried scalars are hoisted out of the
loop into dedicated registers; whatever is left is divided by the peak live
count to decide the unroll factor.

Pure standard library.
"""

import math

from . import isa, vecexp
from .asm import Asm
from .ir import Map, Reduce, topo, contract_program

NVEC = 4                      # float32 lanes per vector register
NREGS = 32                    # v0..v31

# general registers
X_BUFS, X_SARGS, X_OUT, X_TMP = 0, 1, 2, 3
X_BASE0 = 4                   # buffer pointers live in x4 upward
X_COUNT = 16
MAX_BUFS = 12
TANH_COEFFS = [1.0, -0.3333331048488617, 0.13332460820674896,
               -0.053842708468437195, 0.021039675921201706,
               -0.0062355599366128445]
TANH_CROSS = 0.55

KAHAN_THRESHOLD = 4096    # terms per accumulator lane before drift shows


class Schedule:
    """The knobs the autotuner turns."""

    __slots__ = ("unroll", "prefetch", "compensate")

    def __init__(self, unroll=4, prefetch=0, compensate="auto"):
        self.unroll = unroll
        self.prefetch = prefetch      # bytes ahead, 0 = off
        # True / False / "auto". Compensated summation is 92x more accurate
        # at 16M elements and costs 2.9x the time there; at 1M it buys
        # nothing measurable. "auto" turns it on only once each accumulator
        # lane would be summing more than KAHAN_THRESHOLD terms, which is
        # where plain accumulation starts to drift. Both numbers are
        # measured, in bench/reduction_tradeoff.py.
        self.compensate = compensate

    def key(self):
        return (self.unroll, self.prefetch, self.compensate)

    def __repr__(self):
        return (f"Schedule(unroll={self.unroll}, prefetch={self.prefetch}, "
                f"compensate={self.compensate})")


class CompileError(Exception):
    pass


# --------------------------------------------------------------- analysis

def analyse(exprs):
    """Topological order, hoisted leaves, peak live count."""
    order = [n for n in topo(*exprs) if n.op not in ("const", "sarg")]
    pos = {n.id: i for i, n in enumerate(order)}

    lastuse = {n.id: -1 for n in order}
    for i, n in enumerate(order):
        for arg in n.args:
            if arg.id in lastuse:
                lastuse[arg.id] = i
    for e in exprs:
        if e.id in lastuse:
            lastuse[e.id] = len(order)      # outputs live to the end

    live, peak = 0, 0
    for i, n in enumerate(order):
        live += 1
        peak = max(peak, live)
        for arg in n.args:
            if arg.id in lastuse and lastuse[arg.id] == i:
                live -= 1
    return order, pos, lastuse, peak


def constants_of(exprs):
    """Distinct constant values, including the ones exp() needs."""
    vals, has_exp = [], False
    for e in exprs:
        for n in topo(e):
            if n.op == "const" and n.attr not in vals:
                vals.append(n.attr)
            if n.op == "exp":
                has_exp = True
    if has_exp:
        for v in vecexp.CONSTS:
            if v not in vals:
                vals.append(v)
    if any(n.op == "step" for e in exprs for n in topo(e)):
        for v in (0.0, 1.0):
            if v not in vals:
                vals.append(v)
    if any(n.op == "tanh" for e in exprs for n in topo(e)):
        for v in vecexp.CONSTS + TANH_COEFFS + [TANH_CROSS, 1.0, 2.0]:
            if v not in vals:
                vals.append(v)
    return vals, has_exp


def sargs_of(exprs):
    names = []
    for e in exprs:
        for n in topo(e):
            if n.op == "sarg" and n.attr not in names:
                names.append(n.attr)
    return names


# ------------------------------------------------------------ register pool

class RegPool:
    def __init__(self, regs):
        self.free = list(regs)
        self.free.reverse()

    def take(self, k=1):
        if len(self.free) < k:
            raise CompileError("out of vector registers")
        return [self.free.pop() for _ in range(k)]

    def give(self, regs):
        for r in regs:
            self.free.append(r)

    def __len__(self):
        return len(self.free)


# ------------------------------------------------------------------- emit

def _emit_node(a, node, dsts, srcs, cregs, sregs, bases, off_bytes, scratch):
    """Emit one IR node for all U unrolled copies (node-major order)."""
    op = node.op
    U = len(dsts)

    if op == "load":
        base = bases[node.attr]
        for u in range(U):
            a(isa.LDR_q(dsts[u], base, off_bytes + u * 16))
        return

    def arg_regs(k):
        arg = node.args[k]
        if arg.op == "const":
            return [cregs[arg.attr]] * U
        if arg.op == "sarg":
            return [sregs[arg.attr]] * U
        return srcs[k]

    if op in ("add", "sub", "mul", "div", "max", "min"):
        fn = {"add": isa.FADD, "sub": isa.FSUB, "mul": isa.FMUL,
              "div": isa.FDIV, "max": isa.FMAX, "min": isa.FMIN}[op]
        A, B = arg_regs(0), arg_regs(1)
        for u in range(U):
            a(fn(dsts[u], A[u], B[u]))
    elif op == "fma":
        A, B, C = arg_regs(0), arg_regs(1), arg_regs(2)
        for u in range(U):
            # FMLA accumulates into its destination, so seed it with c first
            if dsts[u] != C[u]:
                a(isa.MOV_v(dsts[u], C[u]))
            a(isa.FMLA(dsts[u], A[u], B[u]))
    elif op == "neg":
        A = arg_regs(0)
        for u in range(U):
            a(isa.FNEG(dsts[u], A[u]))
    elif op == "abs":
        A = arg_regs(0)
        for u in range(U):
            a(isa.FABS(dsts[u], A[u]))
    elif op == "sqrt":
        A = arg_regs(0)
        for u in range(U):
            a(isa.FSQRT(dsts[u], A[u]))
    elif op == "recip":
        # Newton-Raphson off the reciprocal estimate: two refinements land
        # within a bit of the true quotient and beat FDIV's latency.
        A = arg_regs(0)
        t = scratch[0]
        for u in range(U):
            a(isa.FRECPE(dsts[u], A[u]))
            a(isa.FRECPS(t, dsts[u], A[u]))
            a(isa.FMUL(dsts[u], dsts[u], t))
            a(isa.FRECPS(t, dsts[u], A[u]))
            a(isa.FMUL(dsts[u], dsts[u], t))
            a(isa.FRECPS(t, dsts[u], A[u]))
            a(isa.FMUL(dsts[u], dsts[u], t))
    elif op == "rsqrt":
        A = arg_regs(0)
        t = scratch[0]
        for u in range(U):
            a(isa.FRSQRTE(dsts[u], A[u]))
            for _ in range(3):
                a(isa.FMUL(t, dsts[u], dsts[u]))
                a(isa.FRSQRTS(t, t, A[u]))
                a(isa.FMUL(dsts[u], dsts[u], t))
    elif op == "step":
        # FCMGT leaves all-ones in the lanes that pass; AND that against a
        # vector of 1.0f and the mask becomes the number 1.0. Two
        # instructions, no branch, no select.
        A = arg_regs(0)
        for u in range(U):
            a(isa.FCMGT(dsts[u], A[u], cregs[0.0]))
            a(isa.AND_v(dsts[u], dsts[u], cregs[1.0]))
    elif op == "tanh":
        # Two formulas, blended by a compare-and-select rather than a branch:
        #   |x| <  0.55   x * P(x^2), a minimax polynomial, no cancellation
        #   |x| >= 0.55   (e^2x - 1)/(e^2x + 1), where e^2x is far from 1
        # Both are computed for every lane and one is selected, because four
        # lanes in one register can disagree about which branch they want and
        # there is no way to send them different ways.
        A = arg_regs(0)
        s0, s1, s2, s3, s4 = scratch[:5]
        for u in range(U):
            d, x = dsts[u], A[u]
            a(isa.FMUL(s0, x, x))                          # x^2
            a(isa.MOV_v(d, cregs[TANH_COEFFS[-1]]))
            for k in range(len(TANH_COEFFS) - 2, -1, -1):
                a(isa.MOV_v(s1, cregs[TANH_COEFFS[k]]))
                a(isa.FMLA(s1, d, s0))
                a(isa.MOV_v(d, s1))
            a(isa.FMUL(d, d, x))                           # x * P(x^2)

            a(isa.FMUL(s1, x, cregs[2.0]))
            vecexp.emit(a, s1, s1, cregs, s2, s3, s4)      # e^2x
            a(isa.FSUB(s2, s1, cregs[1.0]))                # numerator
            a(isa.FADD(s3, s1, cregs[1.0]))                # denominator
            a(isa.FRECPE(s4, s3))
            for _ in range(3):
                a(isa.FRECPS(s1, s4, s3))
                a(isa.FMUL(s4, s4, s1))
            a(isa.FMUL(s1, s2, s4))                        # the exp branch

            a(isa.FABS(s2, x))
            a(isa.FCMGT(s2, cregs[TANH_CROSS], s2))        # 1s where |x| < 0.55
            a(isa.BSL(s2, d, s1))
            a(isa.MOV_v(d, s2))
    elif op == "exp":
        A = arg_regs(0)
        for u in range(U):
            vecexp.emit(a, dsts[u], A[u], cregs,
                        scratch[0], scratch[1], scratch[2])
    else:
        raise CompileError(f"no lowering for op {op!r}")


def _emit_body(a, order, lastuse, exprs, U, cregs, sregs, bases,
               off_bytes, pool, outputs=None, accs=None, how=None,
               comps=None):
    """One pass of the fused body over U*4 elements at byte offset off_bytes."""
    regs = {}                    # node id -> list of U registers
    nscratch = _scratch_count(order)
    scratch = pool.take(nscratch) if nscratch else [None] * 5

    for i, node in enumerate(order):
        dsts = pool.take(U)
        srcs = [regs.get(arg.id) for arg in node.args]
        _emit_node(a, node, dsts, srcs, cregs, sregs, bases, off_bytes, scratch)
        regs[node.id] = dsts
        for arg in node.args:
            if arg.id in lastuse and lastuse[arg.id] == i and arg.id in regs:
                pool.give(regs.pop(arg.id))

    if outputs is not None:
        for name, e in outputs:
            base = bases[name]
            for u in range(U):
                a(isa.STR_q(regs[e.id][u], base, off_bytes + u * 16))
    if accs is not None:
        e = exprs[0]
        v = regs[e.id]
        if how == "max":
            for u in range(U):
                a(isa.FMAX(accs[u], accs[u], v[u]))
        elif comps is None:
            for u in range(U):
                a(isa.FADD(accs[u], accs[u], v[u]))
        else:
            # Compensated (Kahan) summation. A plain running sum loses the
            # low bits of every addend once the accumulator outgrows them,
            # and over millions of elements that drift is visible. Kahan
            # keeps the lost part in a second register and feeds it back, so
            # the error stops growing with n. It costs three extra vector
            # instructions per element - which on a kernel this
            # memory-bound is free, because the loads are the bottleneck.
            #
            #   y = v - c ;  t = s + y ;  c = (t - s) - y ;  s = t
            tmp = pool.take(U)
            for u in range(U):
                a(isa.FSUB(v[u], v[u], comps[u]))          # y
                a(isa.MOV_v(tmp[u], accs[u]))              # keep old s
                a(isa.FADD(accs[u], accs[u], v[u]))        # s = s + y
                a(isa.FSUB(comps[u], accs[u], tmp[u]))     # (t - s_old)
                a(isa.FSUB(comps[u], comps[u], v[u]))      # ... - y
            pool.give(tmp)

    for rs in regs.values():
        pool.give(rs)
    if nscratch:
        pool.give(scratch)


TANH_SCRATCH = 5


def _scratch_count(order):
    """How many spare vector registers the expansions need."""
    if any(n.op == "tanh" for n in order):
        return TANH_SCRATCH
    if any(n.op in ("exp", "recip", "rsqrt") for n in order):
        return vecexp.SCRATCH
    return 0


def _needs_scratch(order):
    return _scratch_count(order) > 0


def _peak_regs(order, lastuse, U, need_scratch):
    live, peak = 0, 0
    for i, n in enumerate(order):
        live += U
        peak = max(peak, live)
        for arg in n.args:
            if arg.id in lastuse and lastuse[arg.id] == i:
                live -= U
    return peak + _scratch_count(order)


# ---------------------------------------------------------------- kernels

class Kernel:
    def __init__(self, stage, asm, sched, unroll, meta):
        self.stage = stage
        self.asm = asm
        self.sched = sched
        self.unroll = unroll
        self.meta = meta
        self.code = None

    def listing(self):
        return self.asm.listing()


def _stage_buffers(stage):
    """Only the buffers this stage actually touches - anything else would
    cost a pointer bump every iteration of the hot loop."""
    used = []
    for e in stage.exprs():
        for nd in topo(e):
            if nd.op == "load" and nd.attr not in used:
                used.append(nd.attr)
    reads = list(used)
    if stage.kind == "map":
        for name, _ in stage.outputs:
            if name not in used:
                used.append(name)
        writes = [nm for nm, _ in stage.outputs]
    else:
        writes = []
    return used, reads, writes


def build_stage(stage, buf_index, sched):
    """Compile one stage into an Asm object."""
    exprs = stage.exprs()
    order, pos, lastuse, peak1 = analyse(exprs)
    cvals, has_exp = constants_of(exprs)
    snames = sargs_of(exprs)
    need_scratch = _needs_scratch(order)

    reserved = len(cvals) + len(snames)
    if stage.kind == "reduce":
        pass                      # accumulators counted below
    avail = NREGS - reserved
    if avail < peak1 + _scratch_count(order) + 1:
        raise CompileError(
            f"expression needs more registers than exist "
            f"({reserved} constants + {peak1} live)")

    def pick_unroll(kahan):
        U = 1
        for cand in range(1, sched.unroll + 1):
            need = _peak_regs(order, lastuse, cand, need_scratch)
            if stage.kind == "reduce":
                need += cand * (3 if kahan else 1)
            if need <= avail:
                U = cand
            else:
                break
        return U

    # Decide compensation from how many terms each accumulator lane would
    # end up summing, then re-pick the unroll factor for that decision -
    # compensation needs a second bank of accumulators, so it can lower U.
    is_sum = stage.kind == "reduce" and stage.how == "sum"
    U = pick_unroll(False)
    if sched.compensate == "auto":
        kahan = is_sum and (stage.n / (NVEC * U)) > KAHAN_THRESHOLD
    else:
        kahan = is_sum and bool(sched.compensate)
    if kahan:
        U = pick_unroll(True)

    n = stage.n
    nvec = n // NVEC
    rem = n - nvec * NVEC
    blocks = nvec // U
    leftover = nvec - blocks * U

    a = Asm(f"{stage.kind}_{n}")

    # ---- prologue: hoist buffer pointers, constants and scalars
    bufs_used, reads, writes = _stage_buffers(stage)
    aliased = any(w in reads for w in writes)
    bases = {}
    for name in bufs_used:
        idx = buf_index[name]
        if idx >= MAX_BUFS:
            raise CompileError("too many buffers")
        bases[name] = X_BASE0 + idx
        a(isa.LDR_x(X_BASE0 + idx, X_BUFS, idx * 8))

    nextv = NREGS - 1
    cregs = {}
    for v in cvals:
        if v == -math.inf:
            a.mov_f32(nextv, X_TMP, -3.4028234663852886e38)
        else:
            a.mov_f32(nextv, X_TMP, v)
        cregs[v] = nextv
        nextv -= 1
    sregs = {}
    for i, nm in enumerate(snames):
        a(isa.LDR_s(nextv, X_SARGS, i * 4))
        a(isa.DUP_lane(nextv, nextv, 0))
        sregs[nm] = nextv
        nextv -= 1

    accs = comps = None
    if stage.kind == "reduce":
        accs = [nextv - i for i in range(U)]
        nextv -= U
        for u in range(U):
            if stage.how == "sum":
                a(isa.MOVI_zero(accs[u]))
            else:
                a.mov_f32(accs[u], X_TMP, -3.4028234663852886e38)
        if kahan:
            comps = [nextv - i for i in range(U)]
            nextv -= U
            for u in range(U):
                a(isa.MOVI_zero(comps[u]))

    pool = RegPool(range(0, nextv + 1))

    outputs = stage.outputs if stage.kind == "map" else None
    how = stage.how if stage.kind == "reduce" else None

    # ---- main loop, trip count baked in
    body_start = body_end = 0
    if blocks > 0:
        a.mov_imm(X_COUNT, blocks)
        a.label("loop")
        body_start = len(a)
        if sched.prefetch:
            for name in bufs_used:
                a(isa.PRFM(bases[name], sched.prefetch))
        _emit_body(a, order, lastuse, exprs, U, cregs, sregs, bases, 0,
                   pool, outputs, accs, how, comps)
        for name in bufs_used:
            a(isa.ADD_imm(bases[name], bases[name], U * 16))
        body_end = len(a)
        a(isa.SUBS_imm(X_COUNT, X_COUNT, 1))
        a.bcond("ne", "loop")

    # ---- leftover whole vectors, straight-line
    if leftover:
        _emit_body(a, order, lastuse, exprs, leftover, cregs, sregs, bases, 0,
                   pool, outputs, accs, how,
                   comps[:leftover] if comps else None)

    # ---- ragged tail
    tail_python = 0
    if rem:
        # The overlap trick re-reads elements it may already have written, so
        # it is only valid when no output buffer is also an input.
        if stage.kind == "map" and n >= NVEC and not aliased:
            # Recompute the last whole vector, overlapping the one before it.
            # Safe because every op here is a pure function of its inputs, so
            # the overlapping lanes get written the same values twice.
            # The pointers now sit past element nvec*4, and we want to load
            # from element n-4, which is 16 - rem*4 bytes behind them.
            if leftover:
                for name in bufs_used:
                    a(isa.ADD_imm(bases[name], bases[name], leftover * 16))
            back = 16 - rem * 4
            for name in bufs_used:
                a(isa.SUB_imm(bases[name], bases[name], back))
            _emit_body(a, order, lastuse, exprs, 1, cregs, sregs, bases, 0,
                       pool, outputs, None, None)
        else:
            tail_python = rem

    # ---- reduce epilogue: fold U accumulators, then across lanes
    if stage.kind == "reduce":
        fn = isa.FADD if stage.how == "sum" else isa.FMAX
        if kahan:
            # the pending correction is worth one last application
            for u in range(U):
                a(isa.FSUB(accs[u], accs[u], comps[u]))
        k = U
        while k > 1:
            half = k // 2
            for u in range(half):
                a(fn(accs[u], accs[u], accs[u + half]))
            if k % 2:
                a(fn(accs[0], accs[0], accs[k - 1]))
            k = half
        if stage.how == "sum":
            a(isa.FADDP(accs[0], accs[0], accs[0]))
            a(isa.FADDP_s(accs[0], accs[0]))
        else:
            a(isa.FMAXV(accs[0], accs[0]))
        a(isa.STR_s(accs[0], X_OUT, 0))

    a(isa.RET())

    # Instruction mix of the hot loop, per element. These are the features
    # the cost model is trained on - it never sees the source expression,
    # only what the loop actually costs.
    body = a.insns[body_start:body_end]
    per_elem = float(U * NVEC) if body else 1.0
    mix = {"ld": 0, "st": 0, "alu": 0, "ptr": 0}
    for ins in body:
        op = ins.text.split()[0]
        if op.startswith("ldr") or op.startswith("ld1") or op == "prfm":
            mix["ld"] += 1
        elif op.startswith("str") or op.startswith("st1"):
            mix["st"] += 1
        elif op in ("add", "sub", "subs", "mov") and " x" in ins.text:
            mix["ptr"] += 1
        else:
            mix["alu"] += 1

    meta = {
        "n": n, "unroll": U,
        "body_insns": len(body),
        "ld_per_elem": mix["ld"] / per_elem,
        "st_per_elem": mix["st"] / per_elem,
        "alu_per_elem": mix["alu"] / per_elem,
        "insns_per_elem": len(body) / per_elem, "blocks": blocks, "leftover": leftover,
        "rem": rem, "tail_python": tail_python,
        "consts": len(cvals), "sargs": len(snames),
        "live": peak1, "regs_used": NREGS - len(pool),
        "nodes": len(order), "has_exp": has_exp,
        "insns": len(a),
        "vec_lanes": (blocks * U + leftover) * NVEC,
        "aliased": aliased, "kahan": kahan,
        "sarg_names": snames,
        "buffers": bufs_used,
    }
    return Kernel(stage, a, sched, U, meta)


def build(prog, sched=None, contract=True):
    """Compile a whole Program. Returns a list of Kernels."""
    if contract:
        contract_program(prog)
    sched = sched or Schedule()
    names = prog.buffers()
    buf_index = {nm: i for i, nm in enumerate(names)}
    return [build_stage(st, buf_index, sched) for st in prog.stages], buf_index
