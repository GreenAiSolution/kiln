"""
kiln.exact - float32 arithmetic done exactly, for the reference evaluator.

Checking a compiler against `float(a) * float(b)` in Python proves nothing:
Python computes in double precision and rounds twice, so it disagrees with
the hardware in ways that have nothing to do with the compiler being right
or wrong. The first version of KILN's tests reported an 86-in-1000 mismatch
rate that was entirely the reference's fault.

So every reference operation here computes the exact mathematical result as
a rational number and rounds it to float32 exactly once, ties to even -
which is the definition of what the hardware does. A disagreement after
that is a real disagreement.

Pure standard library.
"""

import math
import struct
from fractions import Fraction

MANT = 24
EMIN = -126
EMAX = 127
SUBNORMAL_SHIFT = 149          # 23 - EMIN


def bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _ge_pow2(num, den, e):
    """num/den >= 2**e, in exact integer arithmetic."""
    if e >= 0:
        return num >= (den << e)
    return (num << (-e)) >= den


def round_f32(fr):
    """Round an exact Fraction to the nearest float32, ties to even."""
    if fr == 0:
        return 0.0
    neg = fr < 0
    if neg:
        fr = -fr
    num, den = fr.numerator, fr.denominator

    e = num.bit_length() - den.bit_length()
    while not _ge_pow2(num, den, e):
        e -= 1
    while _ge_pow2(num, den, e + 1):
        e += 1

    if e < EMIN:
        shift = SUBNORMAL_SHIFT
        exp = -SUBNORMAL_SHIFT
    else:
        shift = MANT - 1 - e
        exp = e - (MANT - 1)

    if shift >= 0:
        N, D = num << shift, den
    else:
        N, D = num, den << (-shift)
    q, r = divmod(N, D)
    twice = 2 * r
    if twice > D or (twice == D and (q & 1)):
        q += 1

    if q >= (1 << MANT):
        q >>= 1
        exp += 1
    if exp + (MANT - 1) > EMAX:
        return -math.inf if neg else math.inf
    v = math.ldexp(q, exp)
    return -v if neg else v


def F(x):
    return Fraction(x)


def add32(a, b):
    return round_f32(F(a) + F(b))


def sub32(a, b):
    return round_f32(F(a) - F(b))


def mul32(a, b):
    return round_f32(F(a) * F(b))


def div32(a, b):
    if b == 0:
        return math.copysign(math.inf, a) * math.copysign(1.0, b) if a else math.nan
    return round_f32(F(a) / F(b))


def fma32(a, b, c):
    """a*b + c with a single rounding - what FMLA actually computes."""
    return round_f32(F(a) * F(b) + F(c))


def _next_f32(x, up=True):
    b = bits(x)
    if x >= 0:
        b = b + 1 if up else b - 1
    else:
        b = b - 1 if up else b + 1
    return struct.unpack("<f", struct.pack("<I", b & 0xFFFFFFFF))[0]


def sqrt32(x):
    """Correctly rounded float32 square root.

    Double's sqrt then a cast rounds twice. Instead: take the double answer
    as a starting guess and check its neighbours by comparing squared
    midpoints against x in exact rational arithmetic.
    """
    if x < 0:
        return math.nan
    if x == 0 or math.isinf(x):
        return x
    r = struct.unpack("<f", struct.pack("<f", math.sqrt(x)))[0]
    X = F(x)
    for _ in range(4):
        lo = _next_f32(r, up=False)
        hi = _next_f32(r, up=True)
        mid_lo = (F(lo) + F(r)) / 2
        mid_hi = (F(r) + F(hi)) / 2
        if X < mid_lo * mid_lo:
            r = lo
        elif X > mid_hi * mid_hi:
            r = hi
        else:
            break
    return r


def ulps_apart(got, want):
    """Distance in representable float32 steps."""
    if got == want:
        return 0
    if math.isnan(got) or math.isnan(want):
        return float("inf")
    gb = struct.unpack("<i", struct.pack("<f", got))[0]
    wb = struct.unpack("<i", struct.pack("<f", want))[0]
    if gb < 0:
        gb = -2147483648 - gb
    if wb < 0:
        wb = -2147483648 - wb
    return abs(gb - wb)
