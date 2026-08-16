"""
Fit the polynomial that KILN's vectorised exp() uses.

Textbook code drops in a Taylor series. A Taylor series is the best
approximation *at one point* and wastes accuracy everywhere else. The
minimax polynomial is the one whose worst error over the whole interval is
as small as possible, and you find it with Remez exchange: guess where the
error peaks, solve for the polynomial that equioscillates at those points,
move the points to the new peaks, repeat.

Run it and it prints the coefficients, then simulates the exact float32
instruction sequence KILN emits and reports the measured error in ULP
against the C library.

Pure standard library.
"""

import math
import struct
import sys

LN2 = math.log(2.0)
HALF_LN2 = LN2 / 2.0


def f32(x):
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def solve(A, b):
    """Gaussian elimination with partial pivoting."""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-300:
            raise ZeroDivisionError("singular")
        M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        for r in range(n):
            if r == c:
                continue
            f = M[r][c] / piv
            if f:
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] for i in range(n)]


def polyval(c, x):
    v = 0.0
    for k in reversed(c):
        v = v * x + k
    return v


def remez(deg, lo, hi, f, iters=60):
    """Minimax fit of a degree-`deg` polynomial to f on [lo,hi],
    minimising *relative* error."""
    m = deg + 2
    # Chebyshev points as the starting guess
    pts = [((lo + hi) / 2) + ((hi - lo) / 2) * math.cos(math.pi * i / (m - 1))
           for i in range(m)]
    pts.sort()
    coeffs = None
    for _ in range(iters):
        A, b = [], []
        for i, x in enumerate(pts):
            row = [x ** k for k in range(deg + 1)]
            row.append(-((-1) ** i) * abs(f(x)))   # relative-error weight
            A.append(row)
            b.append(f(x))
        sol = solve(A, b)
        coeffs = sol[:deg + 1]
        # find the new error extrema on a dense grid
        N = 20000
        errs = []
        for i in range(N + 1):
            x = lo + (hi - lo) * i / N
            errs.append((x, (polyval(coeffs, x) - f(x)) / abs(f(x))))
        newpts, i = [], 1
        while i < len(errs) - 1 and len(newpts) < m:
            x0, e0 = errs[i - 1]
            x1, e1 = errs[i]
            x2, e2 = errs[i + 1]
            if (e1 >= e0 and e1 >= e2) or (e1 <= e0 and e1 <= e2):
                newpts.append(x1)
                i += 2
            else:
                i += 1
        if len(newpts) >= m:
            # keep the m largest-magnitude alternating extrema
            newpts = sorted(newpts)[:m]
            pts = newpts
        else:
            break
    return coeffs


def simulate(x, coeffs, log2e, ln2hi, ln2lo):
    """Bit-exact simulation of the instruction sequence KILN emits."""
    # n = round(x * log2e)
    n = f32(x * log2e)
    n = float(round(n))            # frintn: round half to even
    if n != n or abs(n) > 1e30:
        n = 0.0
    # r = x - n*ln2hi - n*ln2lo   (two fmls, so the reduction keeps its bits)
    r = f32(x - f32(n * ln2hi))
    r = f32(r - f32(n * ln2lo))
    # Horner in float32 with fused multiply-add
    p = coeffs[-1]
    for c in reversed(coeffs[:-1]):
        # float32 x float32 is exact in double, so this models the hardware
        # FMA's single rounding closely enough to choose a degree. The number
        # that gets reported is measured on the real kernel, not here.
        p = f32(p * r + c)
    # scale by 2^n through the exponent field
    bits = struct.unpack("<I", struct.pack("<f", p))[0]
    bits += int(n) << 23
    if bits >= 0xFF000000 or bits < 0:
        return math.inf
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def ulp_err(got, want):
    if want == 0:
        return 0.0
    gb = struct.unpack("<i", struct.pack("<f", got))[0]
    wb = struct.unpack("<i", struct.pack("<f", f32(want)))[0]
    return abs(gb - wb)


def main():
    deg = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    c = remez(deg, -HALF_LN2, HALF_LN2, math.exp)

    log2e = f32(1.0 / LN2)
    # ln2 split so that n*ln2hi is exact in float32 (low bits of hi are zero)
    ln2hi = struct.unpack("<f", struct.pack(
        "<I", struct.unpack("<I", struct.pack("<f", LN2))[0] & 0xFFFFF000))[0]
    ln2lo = f32(LN2 - ln2hi)
    cf = [f32(v) for v in c]

    print(f"degree               : {deg}")
    print("coefficients (float32, ascending):")
    for i, v in enumerate(cf):
        print(f"  c{i} = {v!r}")
    print(f"log2e                : {log2e!r}")
    print(f"ln2_hi               : {ln2hi!r}   (low 12 mantissa bits zero)")
    print(f"ln2_lo               : {ln2lo!r}")

    theo = max(abs((polyval(c, -HALF_LN2 + (LN2) * i / 20000) - math.exp(
        -HALF_LN2 + LN2 * i / 20000)) / math.exp(-HALF_LN2 + LN2 * i / 20000))
        for i in range(20001))
    print(f"minimax rel. error   : {theo:.3e}  "
          f"({theo / 1.1920929e-7:.2f} float32 eps)")

    worst, worst_x, n = 0, 0.0, 200000
    lo, hi = -87.0, 88.0
    exact = 0
    for i in range(n + 1):
        x = f32(lo + (hi - lo) * i / n)
        got = simulate(x, cf, log2e, ln2hi, ln2lo)
        want = math.exp(x)
        e = ulp_err(got, want)
        if e == 0:
            exact += 1
        if e > worst:
            worst, worst_x = e, x
    print()
    print(f"simulated over        : {n + 1} points in [{lo}, {hi}]")
    print(f"max error             : {worst} ULP   (at x = {worst_x!r})")
    print(f"bit-exact vs libm     : {100.0 * exact / (n + 1):.1f}% of points")

    # emit the constants block for codegen to import
    print()
    print("# --- paste into kiln/vecexp.py ---")
    print(f"COEFFS = {cf!r}")
    print(f"LOG2E = {log2e!r}")
    print(f"LN2_HI = {ln2hi!r}")
    print(f"LN2_LO = {ln2lo!r}")


if __name__ == "__main__":
    main()
