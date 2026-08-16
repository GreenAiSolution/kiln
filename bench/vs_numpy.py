"""
KILN vs numpy, measured.

numpy is a fair and serious opponent: its inner loops are hand-written SIMD
and on this machine its reductions and matmul reach Apple's Accelerate. What
it cannot do is fuse. Asked for `(a*b + c)*a - b`, numpy must walk memory
once per operator and allocate a temporary each time. KILN compiles the whole
expression into one loop that touches each element once and allocates
nothing.

So this measures the thing that actually separates a compiler from a library.

Both sides are timed the same way (best of many runs, same buffers, same
sizes), and numpy is given a second, faster form using preallocated `out=`
arrays - the version a numpy expert would write - so the comparison is not
against a strawman.

numpy is used ONLY here and in the tests. Nothing under kiln/ imports it.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                            # noqa: E402

from kiln import ir, jit                                      # noqa: E402
from kiln.lower import Schedule                               # noqa: E402
from kiln.runtime import bench_batched, compile as kcompile   # noqa: E402


# ------------------------------------------------------------------ cases

class Case:
    def __init__(self, name, build, np_naive, np_out, nbuf_in, nbuf_out,
                 flops, note=""):
        self.name = name
        self.build = build
        self.np_naive = np_naive
        self.np_out = np_out
        self.nbuf_in = nbuf_in
        self.nbuf_out = nbuf_out
        self.flops = flops          # float ops per element
        self.note = note


def c_chain():
    def build(p, n):
        a, b, c = ir.load("a"), ir.load("b"), ir.load("c")
        p.map([("out", (a * b + c) * a - b)], n)

    def naive(A, B, C, O):
        return (A * B + C) * A - B

    def viaout(A, B, C, O, t=[None]):
        if t[0] is None or t[0].shape != A.shape:
            t[0] = np.empty_like(A)
        T = t[0]
        np.multiply(A, B, out=T)
        np.add(T, C, out=T)
        np.multiply(T, A, out=O)
        np.subtract(O, B, out=O)
        return O

    return Case("chain", build, naive, viaout, 3, 1, 4,
                "(a*b + c)*a - b   four operators, one pass")


def c_wide():
    def build(p, n):
        a, b, c, d = (ir.load(x) for x in "abcd")
        p.map([("out", (a * b + c * d) * (a - d) + (b * c - a * a))], n)

    def naive(A, B, C, D, O):
        return (A * B + C * D) * (A - D) + (B * C - A * A)

    def viaout(A, B, C, D, O, t=[None, None]):
        if t[0] is None or t[0].shape != A.shape:
            t[0] = np.empty_like(A)
            t[1] = np.empty_like(A)
        T, U = t
        np.multiply(A, B, out=T)
        np.multiply(C, D, out=U)
        np.add(T, U, out=T)
        np.subtract(A, D, out=U)
        np.multiply(T, U, out=T)
        np.multiply(B, C, out=U)
        np.multiply(A, A, out=O)
        np.subtract(U, O, out=U)
        np.add(T, U, out=O)
        return O

    return Case("wide", build, naive, viaout, 4, 1, 9,
                "nine operators over four arrays")


def c_axpy():
    def build(p, n):
        p.map([("out", ir.load("a") * 2.5 + ir.load("b"))], n)

    def naive(A, B, O):
        return A * 2.5 + B

    def viaout(A, B, O):
        np.multiply(A, 2.5, out=O)
        np.add(O, B, out=O)
        return O

    return Case("axpy", build, naive, viaout, 2, 1, 2,
                "a*2.5 + b   the classic memory-bound kernel")


def c_sigmoid():
    def build(p, n):
        p.map([("out", ir.sigmoid(ir.load("a")))], n)

    def naive(A, O):
        return 1.0 / (1.0 + np.exp(-A))

    def viaout(A, O):
        np.negative(A, out=O)
        np.exp(O, out=O)
        np.add(O, 1.0, out=O)
        np.reciprocal(O, out=O)
        return O

    return Case("sigmoid", build, naive, viaout, 1, 1, 12,
                "1/(1+exp(-x))   transcendental, fused")


def c_gelu():
    def build(p, n):
        p.map([("out", ir.gelu(ir.load("a")))], n)

    def naive(A, O):
        inner = 0.7978845608028654 * (A + 0.044715 * A * A * A)
        return A * 0.5 * (np.tanh(inner) + 1.0)

    def viaout(A, O, t=[None]):
        if t[0] is None or t[0].shape != A.shape:
            t[0] = np.empty_like(A)
        T = t[0]
        np.multiply(A, A, out=T)
        np.multiply(T, A, out=T)
        np.multiply(T, 0.044715, out=T)
        np.add(T, A, out=T)
        np.multiply(T, 0.7978845608028654, out=T)
        np.tanh(T, out=T)
        np.add(T, 1.0, out=T)
        np.multiply(T, 0.5, out=T)
        np.multiply(T, A, out=O)
        return O

    return Case("gelu", build, naive, viaout, 1, 1, 30,
                "the activation every transformer uses")


def c_sumsq():
    def build(p, n):
        a, b = ir.load("a"), ir.load("b")
        d = a - b
        p.reduce("s", "sum", d * d, n)

    def naive(A, B):
        d = A - B
        return float(np.sum(d * d))

    def viaout(A, B, t=[None]):
        if t[0] is None or t[0].shape != A.shape:
            t[0] = np.empty_like(A)
        T = t[0]
        np.subtract(A, B, out=T)
        np.multiply(T, T, out=T)
        return float(np.sum(T))

    return Case("sumsq", build, naive, viaout, 2, 0, 3,
                "sum((a-b)^2)   fused reduction, no temporary")


def c_softmax():
    def build(p, n):
        x = ir.load("a")
        p.reduce("m", "max", x, n)
        e = ir.exp(x - ir.sarg("m"))
        p.reduce("s", "sum", e, n)
        p.map([("out", e * ir.recip(ir.sarg("s")))], n)

    def naive(A, O):
        e = np.exp(A - A.max())
        return e / e.sum()

    def viaout(A, O):
        np.subtract(A, A.max(), out=O)
        np.exp(O, out=O)
        np.multiply(O, 1.0 / O.sum(), out=O)
        return O

    return Case("softmax", build, naive, viaout, 1, 1, 26,
                "three passes instead of five")


def c_layernorm():
    def build(p, n):
        x = ir.load("a")
        p.reduce("s1", "sum", x, n)
        mu = ir.sarg("s1") * ir.f32(1.0 / n)
        p.reduce("s2", "sum", (x - mu) * (x - mu), n)
        inv = ir.rsqrt(ir.sarg("s2") * ir.f32(1.0 / n) + 1e-5)
        p.map([("out", (x - mu) * inv)], n)

    def naive(A, O):
        mu = A.mean()
        v = ((A - mu) ** 2).mean()
        return (A - mu) / np.sqrt(v + 1e-5)

    def viaout(A, O):
        mu = A.mean()
        np.subtract(A, mu, out=O)
        v = float(np.dot(O, O)) / A.size
        np.multiply(O, 1.0 / np.sqrt(v + 1e-5), out=O)
        return O

    return Case("layernorm", build, naive, viaout, 1, 1, 8,
                "mean, variance and normalise")


CASES = [c_axpy(), c_chain(), c_wide(), c_sigmoid(), c_gelu(),
         c_sumsq(), c_softmax(), c_layernorm()]

SIZES = [
    (4096, "L1 (16 KB/array)"),
    (65536, "L2 (256 KB/array)"),
    (1 << 20, "L2/SLC (4 MB/array)"),
    (1 << 24, "DRAM (64 MB/array)"),
]


# ------------------------------------------------------------------ driver

def run_case(case, n, sched, seconds):
    p = ir.Program(case.name)
    case.build(p, n)
    names = p.buffers()

    rng = np.random.default_rng(0)
    arrays = {}
    for nm in p.inputs:
        arrays[nm] = (rng.random(n, dtype=np.float32) * 2 - 1).astype(np.float32)
    for nm in p.outputs:
        arrays.setdefault(nm, np.zeros(n, dtype=np.float32))

    t0 = time.perf_counter()
    c = kcompile(p, sched)
    compile_s = time.perf_counter() - t0

    bufs = {}
    for nm in names:
        b = jit.Buf(n)
        b.frombytes(arrays[nm].tobytes())
        bufs[nm] = b
    c.bind(bufs)

    order = [nm for nm in p.inputs] + [nm for nm in p.outputs
                                       if nm not in p.inputs]
    np_args = [arrays[nm] for nm in order]

    r_k = bench_batched(lambda: c.run(), seconds=seconds)
    r_n = bench_batched(lambda: case.np_naive(*np_args), seconds=seconds)
    r_o = bench_batched(lambda: case.np_out(*np_args), seconds=seconds)

    # Accuracy is scored against a float64 evaluation, not against numpy.
    # Both engines are approximating the same real number, and KILN's fused
    # multiply-adds round once where numpy rounds twice - so "differs from
    # numpy" and "is wrong" are different claims and should be measured
    # separately. Errors are normalised by the scale of the result, because
    # a pointwise relative error is meaningless where the answer is near zero.
    f64_args = [a.astype(np.float64) for a in np_args]
    truth = case.np_naive(*f64_args)
    if p.outputs:
        g = np.asarray(bufs[p.outputs[0]].tolist(), dtype=np.float64)
        w = np.asarray(case.np_naive(*np_args), dtype=np.float64)
        truth = np.asarray(truth, dtype=np.float64)
        scale = max(float(np.max(np.abs(truth))), 1e-30)
        relerr = float(np.max(np.abs(g - truth))) / scale
        nperr = float(np.max(np.abs(w - truth))) / scale
    else:
        kv = float(list(c.run().values())[0])
        nv = float(case.np_naive(*np_args))
        tv = float(truth)
        scale = max(abs(tv), 1e-30)
        relerr = abs(kv - tv) / scale
        nperr = abs(nv - tv) / scale

    bytes_moved = n * 4 * (case.nbuf_in + case.nbuf_out)
    if case.name in ("softmax", "layernorm"):
        bytes_moved = n * 4 * 3          # three passes

    return {
        "case": case.name, "n": n,
        "kiln": r_k["best"], "numpy": r_n["best"], "numpy_out": r_o["best"],
        "speedup_naive": r_n["best"] / r_k["best"],
        "speedup_out": r_o["best"] / r_k["best"],
        "kiln_gbs": bytes_moved / r_k["best"] / 1e9,
        "numpy_gbs": bytes_moved / r_n["best"] / 1e9,
        "kiln_gflops": case.flops * n / r_k["best"] / 1e9,
        "unroll": c.kernels[0].meta["unroll"],
        "insns": c.insn_count(),
        "code_bytes": c.code_bytes(),
        "compile_ms": compile_s * 1e3,
        "relerr": relerr,
        "nperr": nperr,
        "note": case.note,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=0.15)
    ap.add_argument("--unroll", type=int, default=6)
    ap.add_argument("--json", default="")
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    sched = Schedule(args.unroll)
    out = []
    print("KILN vs numpy   (Apple M1 Max, float32, single core)")
    print("=" * 92)
    for n, label in SIZES:
        print(f"\nn = {n:,}   {label}")
        print(f"  {'kernel':<11} {'kiln us':>9} {'numpy us':>9} {'np+out us':>10} "
              f"{'vs numpy':>9} {'vs np+out':>10} {'GB/s':>7} {'kiln err':>9} {'numpy err':>10}")
        for case in CASES:
            if args.only and args.only not in case.name:
                continue
            r = run_case(case, n, sched, args.seconds)
            out.append(r)
            print(f"  {r['case']:<11} {r['kiln'] * 1e6:>9.2f} "
                  f"{r['numpy'] * 1e6:>9.2f} {r['numpy_out'] * 1e6:>10.2f} "
                  f"{r['speedup_naive']:>8.2f}x {r['speedup_out']:>9.2f}x "
                  f"{r['kiln_gbs']:>7.1f} {r['relerr']:>9.2e} {r['nperr']:>10.2e}")

    print("\n" + "=" * 92)
    sp = [r["speedup_naive"] for r in out]
    so = [r["speedup_out"] for r in out]
    print(f"speedup vs idiomatic numpy : "
          f"median {sorted(sp)[len(sp) // 2]:.2f}x   best {max(sp):.2f}x   "
          f"worst {min(sp):.2f}x")
    print(f"speedup vs numpy with out= : "
          f"median {sorted(so)[len(so) // 2]:.2f}x   best {max(so):.2f}x   "
          f"worst {min(so):.2f}x")
    wk = max(r['relerr'] for r in out)
    wn = max(r['nperr'] for r in out)
    closer = sum(1 for r in out if r['relerr'] <= r['nperr'])
    print(f"worst error vs float64     : kiln {wk:.3e}   numpy {wn:.3e}")
    print(f"cases where kiln is at least as accurate as numpy: "
          f"{closer}/{len(out)}")
    print(f"compile time               : "
          f"{sum(r['compile_ms'] for r in out) / len(out):.2f} ms average")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
