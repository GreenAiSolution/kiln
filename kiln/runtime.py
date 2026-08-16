"""
kiln.runtime - compile a Program and call the result.

Holds the JIT'd code alive, builds the pointer table the kernels read their
buffer addresses from, and finishes any ragged tail of fewer than four
elements in Python. The tail path uses the same reference evaluator the
tests check against, so a size of 1000 and a size of 1001 give the same
answers to the bit.

Pure standard library.
"""

import ctypes
import math
import time

from . import ir, jit, lower
from .lower import Schedule


class Compiled:
    def __init__(self, prog, sched=None, contract=True):
        self.prog = prog
        self.sched = sched or Schedule()
        self.kernels, self.buf_index = lower.build(prog, self.sched, contract)
        self.codes = [jit.load(k.asm) for k in self.kernels]
        self.fns = [c.fn(None, [jit.VOIDP, jit.VOIDP, jit.VOIDP])
                    for c in self.codes]
        self.names = prog.buffers()
        self._table = (ctypes.c_void_p * max(1, len(self.names)))()
        self._sargbuf = (ctypes.c_float * 16)()
        self._out = (ctypes.c_float * 1)()
        self._bound = None
        # Cast once. Doing this per call costs about a microsecond, which is
        # most of the runtime for a small kernel.
        self._tp = ctypes.cast(self._table, ctypes.c_void_p)
        self._sp = ctypes.cast(self._sargbuf, ctypes.c_void_p)
        self._op = ctypes.cast(self._out, ctypes.c_void_p)
        # A single map stage with no scalars and no Python tail is by far the
        # common case; it gets a path with no per-call bookkeeping at all.
        k = self.kernels[0]
        self._simple = (len(self.kernels) == 1
                        and k.stage.kind == "map"
                        and not k.meta["sarg_names"]
                        and not k.meta["tail_python"])
        self._plan = list(zip(self.kernels, self.fns))

    # ------------------------------------------------------------- calling

    def bind(self, bufs):
        """Point the table at these buffers. Call once, then run many times."""
        for i, nm in enumerate(self.names):
            b = bufs[nm]
            self._table[i] = b.ptr
        self._bound = bufs
        return self

    def run(self, bufs=None):
        if bufs is not None:
            self.bind(bufs)
        if self._simple:
            self.fns[0](self._tp, self._sp, self._op)
            return {}
        bufs = self._bound
        scalars = {}
        for k, fn in self._plan:
            st = k.stage
            for i, nm in enumerate(k.meta["sarg_names"]):
                self._sargbuf[i] = scalars[nm]
            fn(self._tp, self._sp, self._op)
            if st.kind == "reduce":
                acc = self._out[0]
                done = k.meta["vec_lanes"]
                for i in range(done, st.n):
                    v = ir.eval_expr(st.expr, i, bufs, scalars, {})
                    acc = ir.f32(acc + v) if st.how == "sum" else max(acc, v)
                scalars[st.name] = acc
            elif k.meta["tail_python"]:
                done = k.meta["vec_lanes"]
                for i in range(done, st.n):
                    cache = {}
                    for nm, e in st.outputs:
                        bufs[nm][i] = ir.eval_expr(e, i, bufs, scalars, cache)
        return scalars

    __call__ = run

    # ------------------------------------------------------------ reporting

    def code_bytes(self):
        return sum(c.nbytes for c in self.codes)

    def insn_count(self):
        return sum(k.meta["insns"] for k in self.kernels)

    def report(self):
        rows = []
        for k in self.kernels:
            m = k.meta
            rows.append({
                "stage": k.stage.kind,
                "n": m["n"],
                "nodes": m["nodes"],
                "unroll": m["unroll"],
                "live": m["live"],
                "consts": m["consts"],
                "regs": m["regs_used"],
                "insns": m["insns"],
                "loop_trip": m["blocks"],
            })
        return rows

    def listing(self, i=0):
        return self.kernels[i].listing()


def compile(prog, sched=None, contract=True):
    return Compiled(prog, sched, contract)


# ------------------------------------------------------------- measurement

_spun = [False]


def spin_up(seconds=0.35):
    """Get the core to its full clock before timing anything.

    Apple Silicon ramps frequency on demand, and the first thing measured in
    a fresh process runs on a core that has not ramped yet. That made the
    first benchmark in every run read low - it is how a peak-throughput
    kernel came out at 3.32 fused multiply-adds per cycle when the real
    number is 4.00. Measured once and cached, because it only has to happen
    once per process.
    """
    if _spun[0]:
        return
    t_end = time.perf_counter() + seconds
    x = 0.0
    while time.perf_counter() < t_end:
        for _ in range(2000):
            x = x * 1.0000001 + 1.0
    _spun[0] = bool(x) or True


def bench(fn, seconds=0.25, min_reps=3, warmup=2):
    """Best-of timing. The minimum is the right statistic here: interference
    from other processes only ever makes a run slower, never faster."""
    spin_up()
    for _ in range(warmup):
        fn()
    reps, best, total = 0, math.inf, 0.0
    t_end = time.perf_counter() + seconds
    while reps < min_reps or time.perf_counter() < t_end:
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        best = min(best, dt)
        total += dt
        reps += 1
        if reps > 200000:
            break
    return {"best": best, "mean": total / reps, "reps": reps}


def bench_batched(fn, inner=None, seconds=0.25):
    """For kernels so fast that one call is below timer resolution: run a
    batch and divide."""
    if inner is None:
        t0 = time.perf_counter()
        fn()
        one = time.perf_counter() - t0
        inner = max(1, int(2e-4 / max(one, 1e-9)))
    def batch():
        for _ in range(inner):
            fn()
    r = bench(batch, seconds=seconds)
    return {"best": r["best"] / inner, "mean": r["mean"] / inner,
            "reps": r["reps"] * inner, "inner": inner}
