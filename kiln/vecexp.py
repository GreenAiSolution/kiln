"""
kiln.vecexp - exp() for four float32 lanes at once, in eleven instructions.

There is no vector exp instruction on ARM, and calling libm would break the
loop and serialise everything, so KILN emits its own:

    x   = clamp(x, -87.3, 88.7)        keep the result finite
    n   = round(x * log2(e))           how many powers of two
    r   = x - n*ln2_hi - n*ln2_lo      remainder in [-ln2/2, ln2/2],
                                       split in two so the subtraction is exact
    p   = poly6(r)                     minimax, not Taylor - see tools/fit_exp.py
    out = p with n added to its exponent field

The last step is the trick: multiplying by 2^n is just adding n to the
exponent bits, so it costs an integer shift and an integer add instead of a
multiply and a table lookup.

Coefficients come from tools/fit_exp.py (Remez exchange, degree 6).
Measured accuracy is in tests/verify_exp.py - not asserted here.
"""

from . import isa

COEFFS = [1.0, 1.0, 0.49999991059303284, 0.16666418313980103,
          0.041668206453323364, 0.008374955505132675, 0.0013838205486536026]
LOG2E = 1.4426950216293335
LN2_HI = 0.693115234375
LN2_LO = 3.194618329871446e-05
CLAMP_LO = -87.3
CLAMP_HI = 88.7

# Constant values the expansion needs a register for.
CONSTS = [CLAMP_LO, CLAMP_HI, LOG2E, LN2_HI, LN2_LO] + COEFFS

SCRATCH = 3          # vector registers needed beyond the destination
INSN_COUNT = 6 + 2 * len(COEFFS) - 2 + 3


def emit(a, dst, src, cregs, t0, t1, t2):
    """Emit exp(src) -> dst.

    cregs maps a constant value to the vector register holding it.
    t0, t1, t2 are scratch vector registers, clobbered.
    dst may alias src.
    """
    c = cregs
    a(isa.FMAX(dst, src, c[CLAMP_LO]))
    a(isa.FMIN(dst, dst, c[CLAMP_HI]))

    # n = round-to-nearest(x * log2e), kept as a float for the FMLS steps
    a(isa.FMUL(t0, dst, c[LOG2E]))
    a(isa.FRINTN(t0, t0))

    # r = x - n*ln2_hi - n*ln2_lo, in place in dst
    a(isa.FMLS(dst, t0, c[LN2_HI]))
    a(isa.FMLS(dst, t0, c[LN2_LO]))

    # Horner, ping-ponging between t1 and t2 because FMLA accumulates into
    # its destination: t_new = c_k + p*r
    cur, nxt = t1, t2
    a(isa.MOV_v(cur, c[COEFFS[-1]]))
    for k in range(len(COEFFS) - 2, -1, -1):
        a(isa.MOV_v(nxt, c[COEFFS[k]]))
        a(isa.FMLA(nxt, cur, dst))
        cur, nxt = nxt, cur

    # multiply by 2^n by adding n to the exponent field
    a(isa.FCVTZS(t0, t0))
    a(isa.VSHL(t0, t0, 23))
    a(isa.VADD(dst, cur, t0))
