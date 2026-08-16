"""
Fit the polynomial for the small-argument branch of tanh.

tanh(x) = (e^2x - 1)/(e^2x + 1) is accurate for large |x| and terrible for
small |x|: e^2x approaches 1, so the numerator subtracts two nearly equal
numbers and most of the significant digits cancel. Measured, that formula
reaches 254 ULP near zero.

tanh is odd, so near zero it is x times an even function of x. Fitting
tanh(x)/x as a minimax polynomial in x^2 has no cancellation at all, and the
two branches get blended with a compare-and-select instead of a branch.

Prints the coefficients and the crossover point's error on both sides.
"""

import math
import sys
sys.path.insert(0, __file__.rsplit("/", 2)[0])
from tools.fit_exp import f32, remez, polyval          # noqa: E402

CROSS = 0.55


def main():
    deg = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    hi = CROSS * CROSS

    def g(y):
        x = math.sqrt(y)
        return math.tanh(x) / x if x > 1e-12 else 1.0

    c = remez(deg, 1e-9, hi, g)
    cf = [f32(v) for v in c]
    print(f"branch point      : |x| < {CROSS}")
    print(f"degree in x^2     : {deg}")
    for i, v in enumerate(cf):
        print(f"  t{i} = {v!r}")

    worst = 0.0
    N = 40000
    for i in range(N + 1):
        y = hi * i / N
        e = abs((polyval(cf, y) - g(y)) / g(y))
        worst = max(worst, e)
    print(f"poly rel error    : {worst:.3e}  ({worst / 1.1920929e-7:.2f} eps)")

    # the exp branch, at the crossover, is where it is weakest
    u = math.exp(2 * CROSS)
    rel = 1.1920929e-7 * u / (u - 1)
    print(f"exp branch at {CROSS} : ~{rel / 1.1920929e-7:.1f} eps "
          f"(cancellation in e^2x - 1)")
    print()
    print("# --- paste into kiln/lower.py ---")
    print(f"TANH_COEFFS = {cf!r}")
    print(f"TANH_CROSS = {CROSS!r}")


if __name__ == "__main__":
    main()
