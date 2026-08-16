"""
How accurate is KILN's vectorised exp, actually?

tools/fit_exp.py predicted 1 ULP from a simulation. This runs the real
kernel, on the real hardware, over a dense sweep of the whole useful domain,
and compares every result against the C library.

Reference caveat, stated rather than buried: the comparison target is the
double-precision exp rounded to float32. That is the correctly rounded
float32 answer except where the exact value sits within about 2^-29 of a
float32 midpoint, which is rare enough not to move a maximum-ULP number but
is not the same as a proof.
"""

import math
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kiln import exact, ir, jit, vecexp                       # noqa: E402
from kiln.lower import Schedule                               # noqa: E402
from kiln.runtime import bench_batched, compile as kcompile   # noqa: E402


def f32(x):
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def sweep(lo, hi, n):
    return [f32(lo + (hi - lo) * i / (n - 1)) for i in range(n)]


def run(xs):
    n = len(xs)
    n4 = ((n + 3) // 4) * 4
    p = ir.Program("exp")
    p.map([("out", ir.exp(ir.load("a")))], n4)
    ir.contract_program(p)
    a = jit.Buf(n4)
    for i, v in enumerate(xs):
        a[i] = v
    out = jit.Buf(n4)
    kcompile(p, Schedule(6), contract=False).run({"a": a, "out": out})
    return out.tolist()[:n]


def report(name, lo, hi, n):
    xs = sweep(lo, hi, n)
    got = run(xs)
    worst, worst_x, exactcnt, hist = 0, 0.0, 0, {}
    for x, g in zip(xs, got):
        want = f32(math.exp(x))
        u = exact.ulps_apart(g, want)
        hist[u] = hist.get(u, 0) + 1
        if u == 0:
            exactcnt += 1
        if u > worst:
            worst, worst_x = u, x
    dist = " ".join(f"{k}:{100.0 * v / n:.1f}%" for k, v in sorted(hist.items())
                    if k <= 3)
    print(f"  {name:<26} {n:>8} pts   max {worst} ULP   "
          f"bit-exact {100.0 * exactcnt / n:5.1f}%   [{dist}]")
    return worst


def main():
    print("KILN vectorised exp - measured on hardware")
    print("=" * 78)
    print(f"polynomial degree : {len(vecexp.COEFFS) - 1} "
          f"(minimax, from tools/fit_exp.py)")
    print(f"instructions      : {vecexp.INSN_COUNT} per four lanes, "
          f"{vecexp.SCRATCH} scratch registers")
    print(f"constants         : {len(vecexp.CONSTS)} vector registers")
    print()
    worst = 0
    worst = max(worst, report("full domain", -87.0, 88.0, 200003))
    worst = max(worst, report("neural net range", -12.0, 12.0, 200003))
    worst = max(worst, report("softmax range", -30.0, 0.0, 200003))
    worst = max(worst, report("near zero", -1e-3, 1e-3, 100003))
    worst = max(worst, report("near overflow", 80.0, 88.7, 100003))
    worst = max(worst, report("near underflow", -87.3, -80.0, 100003))

    # throughput against libm, one element at a time in Python is not a fair
    # fight, so compare the whole-array cost
    n = 1 << 20
    p = ir.Program("exp")
    p.map([("out", ir.exp(ir.load("a")))], n)
    ir.contract_program(p)
    a, out = jit.Buf(n, 0.5), jit.Buf(n)
    c = kcompile(p, Schedule(10), contract=False)
    c.bind({"a": a, "out": out})
    r = bench_batched(lambda: c.run(), seconds=0.3)
    print()
    print(f"throughput        : {n / r['best'] / 1e9:.2f} billion exp/s "
          f"({r['best'] / n * 1e9:.2f} ns each, one core)")

    print()
    ok = worst <= 2
    print(f"worst over all sweeps: {worst} ULP  -> "
          f"{'PASS' if ok else 'FAIL'} (bound 2)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
