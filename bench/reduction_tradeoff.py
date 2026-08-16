"""
The reduction accuracy/speed trade-off, measured instead of asserted.

A plain running sum loses the low bits of each addend once the accumulator
outgrows them. The error is a random walk in the number of terms each
accumulator lane sees, so it gets worse as the array gets bigger and there
is no way to unroll around it.

Compensated (Kahan) summation stops the drift by carrying the lost part in a
second register. It costs three extra vector instructions per element, and
on a reduction that is already latency-bound that is not free - it is about
3x. So KILN does not simply pick one. It turns compensation on once each
accumulator lane would be summing more than KAHAN_THRESHOLD terms.

This script prints the numbers behind that rule: error and time for both
settings, against the exact sum and against numpy.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                            # noqa: E402

from kiln import ir, jit, lower                               # noqa: E402
from kiln.lower import Schedule                               # noqa: E402
from kiln.runtime import bench_batched, compile as kcompile   # noqa: E402

EPS32 = 1.1920928955078125e-07


def build(p, n):
    a, b = ir.load("a"), ir.load("b")
    d = a - b
    p.reduce("s", "sum", d * d, n)


def main():
    print("Reduction: what compensation buys and what it costs")
    print("=" * 92)
    print(f"KAHAN_THRESHOLD = {lower.KAHAN_THRESHOLD} terms per accumulator lane")
    print()
    print(f"  {'n':>10} {'terms/lane':>11} | {'plain err':>11} {'plain us':>10}"
          f" | {'kahan err':>11} {'kahan us':>10} | {'numpy err':>11} "
          f"{'numpy us':>9} | {'auto':>6}")
    print("-" * 92)
    rng = np.random.default_rng(1)
    for n in (1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22, 1 << 24):
        A = rng.random(n, dtype=np.float32).astype(np.float32)
        B = (rng.random(n, dtype=np.float32) * 0.5).astype(np.float32)
        truth = math.fsum((float(x) - float(y)) ** 2 for x, y in zip(A, B))

        bufs = {}
        for nm, arr in (("a", A), ("b", B)):
            buf = jit.Buf(n)
            buf.frombytes(arr.tobytes())
            bufs[nm] = buf

        row = {}
        for tag, comp in (("plain", False), ("kahan", True), ("auto", "auto")):
            p = ir.Program("sumsq")
            build(p, n)
            ir.contract_program(p)
            c = kcompile(p, Schedule(6, compensate=comp), contract=False)
            c.bind(bufs)
            got = c.run()["s"]
            t = bench_batched(lambda: c.run(), seconds=0.15)["best"]
            row[tag] = (abs(got - truth) / truth, t,
                        c.kernels[0].meta["kahan"],
                        c.kernels[0].meta["unroll"])

        d = A - B
        npv = float(np.sum(d * d))
        tn = bench_batched(lambda: float(np.sum((A - B) ** 2)), seconds=0.15)["best"]
        nperr = abs(npv - truth) / truth

        U = row["plain"][3]
        per_lane = n / (4 * U)
        chose = "kahan" if row["auto"][2] else "plain"
        print(f"  {n:>10} {per_lane:>11.0f} | {row['plain'][0]:>11.2e} "
              f"{row['plain'][1] * 1e6:>10.1f} | {row['kahan'][0]:>11.2e} "
              f"{row['kahan'][1] * 1e6:>10.1f} | {nperr:>11.2e} {tn * 1e6:>9.1f} "
              f"| {chose:>6}")

    print()
    print("Reading this: at small sizes plain summation is already accurate and")
    print("compensation is pure cost. At large sizes plain drifts by orders of")
    print("magnitude and compensation is the only thing keeping the answer")
    print("meaningful. The threshold is where those two curves cross.")


if __name__ == "__main__":
    main()
