"""
Numerical verification of every kernel KILN generates.

Two standards, and it matters which applies where:

  EXACT   - for kernels built only from add/sub/mul/fma/max/min/sqrt, the
            machine code must agree with the exact-rational reference to the
            last bit, for every element. No tolerance. Any single differing
            bit is a failure.

  MEASURED - exp, recip and rsqrt are approximations by construction (a
            polynomial, and Newton-Raphson off a hardware estimate). There is
            no "correct" bit pattern to demand, so the test reports the worst
            error in ULP and fails only past a stated bound.

Reductions are checked against math.fsum, which is exact, and the report
shows both KILN's error and a plain sequential float32 loop's error, because
KILN's tree reduction is the more accurate of the two and that should be
visible rather than asserted.
"""

import math
import os
import random
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kiln import exact, ir, jit                              # noqa: E402
from kiln.lower import Schedule                              # noqa: E402
from kiln.runtime import compile as kcompile                 # noqa: E402

SIZES = (4, 5, 7, 16, 17, 63, 256, 1000, 1024, 4099, 65536)
SCHEDULES = (Schedule(1), Schedule(2), Schedule(3), Schedule(4),
             Schedule(6), Schedule(8), Schedule(4, prefetch=256))

FAILS = []
ROWS = []


def seed_of(*parts):
    """A stable seed. Python's hash() of a string is randomised per process,
    so using it here made the test data change between runs - which is how a
    254 ULP error in tanh went unnoticed for several runs and then failed one.
    A test whose inputs move is not a test."""
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7FFFFFFF


def rnd(n, seed, lo=-2.0, hi=2.0):
    r = random.Random(seed)
    return [ir.f32(r.uniform(lo, hi)) for _ in range(n)]


EPS32 = 1.1920928955078125e-07


def sum_error_budget(n):
    """What a *correct* float32 summation of n terms may be off by, relative
    to the sum of absolute terms. Any ordering satisfies c*eps with
    c <= n; a pairwise/tree ordering satisfies c ~ log2(n). We allow a
    generous 8*log2(n) - still far tighter than the n*eps a sequential loop
    would need, so this test would catch a broken reduction."""
    return 8.0 * max(1.0, math.log2(max(n, 2))) * EPS32


def run_case(name, build, n, sched, tol_ulp=0, lo=-2.0, hi=2.0,
             tol_abs=None):
    p = ir.Program(name)
    build(p, n)
    ir.contract_program(p)          # the reference must model what we compile

    data = {}
    for i, nm in enumerate(p.inputs):
        data[nm] = rnd(n, seed_of(name, nm, n, i), lo, hi)
    for nm in p.outputs:
        data.setdefault(nm, [0.0] * n)

    bufs = {k: jit.Buf.of(v) for k, v in data.items()}
    c = kcompile(p, sched, contract=False)
    got_scalars = c.run(bufs)

    # Replay the reference on the same inputs, but feeding it KILN's own
    # scalars, so each stage is scored on exactly what the machine code saw.
    ref = {k: list(v) for k, v in data.items()}
    ideal, cond = ir.run_reference(p, ref, use_scalars=got_scalars)

    worst, nbad, worst_abs = 0, 0, 0.0
    for nm in p.outputs:
        g, w = bufs[nm].tolist(), ref[nm]
        scale = max((abs(v) for v in w), default=1.0) or 1.0
        for x, y in zip(g, w):
            u = exact.ulps_apart(x, y)
            if u:
                nbad += 1
                worst = max(worst, u)
            worst_abs = max(worst_abs, abs(x - y) / scale)
    # gelu's tanh form computes 1 + tanh(z), and where tanh(z) -> -1 that
    # subtraction destroys every significant digit. The output there is about
    # 1e-7 while the function's range is 8, so a ULP count is meaningless and
    # an error relative to the function's scale is the honest measure. The
    # cancellation is in the formula transformers use, not in the compiler.
    ok = (worst_abs <= tol_abs) if tol_abs is not None else (worst <= tol_ulp)

    # Reductions are judged against the exact sum with a backward-error bound,
    # not against some other arbitrary summation order.
    red_err = 0.0
    for nm in p.scalars:
        scale = cond.get(nm, 0.0)
        err = abs(got_scalars[nm] - ideal[nm])
        rel = err / scale if scale else 0.0
        red_err = max(red_err, rel / EPS32)
        if rel > sum_error_budget(n):
            ok = False
            FAILS.append((name, n, sched.key(), f"reduce rel={rel:.2e}"))

    ROWS.append((name, n, sched.key(), c.kernels[0].meta["unroll"],
                 c.insn_count(), nbad, worst, ok, red_err, worst_abs,
                 tol_abs is not None))
    if tol_abs is not None:
        if worst_abs > tol_abs:
            FAILS.append((name, n, sched.key(),
                          f"{worst_abs:.2e} relative > {tol_abs:.0e}"))
    elif worst > tol_ulp:
        FAILS.append((name, n, sched.key(), f"{worst} ULP > {tol_ulp}"))
    return ok


# ------------------------------------------------------------------ kernels

def k_chain(p, n):
    a, b, c = ir.load("a"), ir.load("b"), ir.load("c")
    p.map([("out", (a * b + c) * a - b)], n)


def k_axpy(p, n):
    p.map([("out", ir.load("a") * 2.5 + ir.load("b"))], n)


def k_poly(p, n):
    x = ir.load("a")
    p.map([("out", ((((x * 0.5 + 1.5) * x - 2.25) * x + 3.125) * x - 0.75))], n)


def k_relu(p, n):
    a, b = ir.load("a"), ir.load("b")
    p.map([("out", ir.relu(a * 2.0 + b) * ir.relu(b - a))], n)


def k_norm(p, n):
    a, b = ir.load("a"), ir.load("b")
    p.map([("out", ir.sqrt(a * a + b * b))], n)


def k_multi_out(p, n):
    a, b = ir.load("a"), ir.load("b")
    s = a + b
    p.map([("o1", s * s), ("o2", s - a * b), ("o3", ir.relu(s))], n)


def k_wide(p, n):
    a, b, c, d = (ir.load(x) for x in "abcd")
    p.map([("out", (a * b + c * d) * (a - d) + (b * c - a * a))], n)


def k_exp(p, n):
    a = ir.load("a")
    p.map([("out", ir.exp(a))], n)


def k_sigmoid(p, n):
    p.map([("out", ir.sigmoid(ir.load("a")))], n)


def k_tanh(p, n):
    p.map([("out", ir.tanh(ir.load("a")))], n)


def k_gelu(p, n):
    p.map([("out", ir.gelu(ir.load("a")))], n)


def k_recip(p, n):
    p.map([("out", ir.recip(ir.load("a")))], n)


def k_rsqrt(p, n):
    p.map([("out", ir.rsqrt(ir.load("a")))], n)


def k_sum(p, n):
    p.reduce("s", "sum", ir.load("a"), n)


def k_sumsq(p, n):
    a, b = ir.load("a"), ir.load("b")
    d = a - b
    p.reduce("s", "sum", d * d, n)


def k_max(p, n):
    p.reduce("m", "max", ir.load("a"), n)


def k_softmax(p, n):
    x = ir.load("a")
    p.reduce("m", "max", x, n)
    e = ir.exp(x - ir.sarg("m"))
    p.reduce("s", "sum", e, n)
    p.map([("out", e * ir.recip(ir.sarg("s")))], n)


def k_layernorm(p, n):
    x = ir.load("a")
    p.reduce("s1", "sum", x, n)
    mu = ir.sarg("s1") * ir.f32(1.0 / n)
    p.reduce("s2", "sum", (x - mu) * (x - mu), n)
    inv = ir.rsqrt(ir.sarg("s2") * ir.f32(1.0 / n) + 1e-5)
    p.map([("out", (x - mu) * inv)], n)


EXACT_KERNELS = [
    ("chain", k_chain), ("axpy", k_axpy), ("poly5", k_poly),
    ("relu", k_relu), ("hypot", k_norm), ("multi_out", k_multi_out),
    ("wide", k_wide),
]
# (name, builder, ULP bound, or relative-to-scale bound where ULP does not apply)
APPROX_KERNELS = [
    ("exp", k_exp, 3, None), ("sigmoid", k_sigmoid, 4, None),
    ("tanh", k_tanh, 4, None), ("recip", k_recip, 2, None),
    ("rsqrt", k_rsqrt, 3, None),
    ("gelu", k_gelu, None, 1e-6),
]


def reduction_accuracy():
    """KILN's reduction tree vs a sequential float32 loop, both against the
    exact sum."""
    print("\nreduction accuracy (vs exact sum)")
    print(f"  {'n':>8}  {'kiln err':>12}  {'sequential err':>16}  {'ratio':>8}")
    out = []
    for n in (1024, 16384, 262144, 1048576):
        vals = rnd(n, 7, 0.0, 1.0)
        p = ir.Program("sum")
        p.reduce("s", "sum", ir.load("a"), n)
        bufs = {"a": jit.Buf.of(vals)}
        c = kcompile(p, Schedule(8))
        got = c.run(bufs)["s"]
        truth = math.fsum(vals)
        seq = 0.0
        for v in vals:
            seq = exact.add32(seq, v)
        ek = abs(got - truth) / abs(truth)
        es = abs(seq - truth) / abs(truth)
        ratio = es / ek if ek else float("inf")
        print(f"  {n:>8}  {ek:>12.3e}  {es:>16.3e}  {ratio:>7.1f}x")
        out.append((n, ek, es))
    return out


def main():
    print("KILN kernel verification")
    print("=" * 74)
    for sched in SCHEDULES:
        for n in SIZES:
            for name, fn in EXACT_KERNELS:
                run_case(name, fn, n, sched, tol_ulp=0)
            for name, fn, tol, tabs in APPROX_KERNELS:
                lo, hi = (0.05, 4.0) if name in ("recip", "rsqrt") else (-6.0, 6.0)
                run_case(name, fn, n, sched, tol_ulp=tol or 0, lo=lo, hi=hi,
                         tol_abs=tabs)
            run_case("sum", k_sum, n, sched, tol_ulp=64)
            run_case("sumsq", k_sumsq, n, sched, tol_ulp=64)
            run_case("max", k_max, n, sched, tol_ulp=0)
            run_case("softmax", k_softmax, n, sched, tol_ulp=8)
            run_case("layernorm", k_layernorm, n, sched, tol_ulp=64)

    # per-kernel worst case across every size and schedule
    agg = {}
    for name, n, sk, U, insns, nbad, worst, ok, red, wabs, isabs in ROWS:
        cur = agg.setdefault(name, [0, 0, 0, 0.0, 0.0, isabs])
        cur[0] = max(cur[0], worst)
        cur[1] += 1
        cur[2] += 0 if ok else 1
        cur[3] = max(cur[3], red)
        cur[4] = max(cur[4], wabs)

    print(f"\n{'kernel':<12} {'cases':>6} {'worst ULP':>10} {'rel to scale':>13} "
          f"{'reduce err':>11} {'failed':>7}   standard")
    print("-" * 88)
    exact_names = {n for n, _ in EXACT_KERNELS} | {"max"}
    for name in sorted(agg):
        worst, cases, failed, red, wabs, isabs = agg[name]
        std = ("EXACT" if name in exact_names
               else "rel. to scale" if isabs else "ULP")
        flag = "" if failed == 0 else "  <-- FAIL"
        redtxt = f"{red:.1f} eps" if red else "-"
        print(f"{name:<12} {cases:>6} {worst:>10} {wabs:>13.2e} {redtxt:>11} "
              f"{failed:>7}   {std}{flag}")

    total = len(ROWS)
    elems = sum(r[1] for r in ROWS)
    print("-" * 74)
    print(f"{total} kernels compiled and checked, "
          f"{elems:,} elements compared element by element")
    print(f"schedules: {len(SCHEDULES)}   sizes: {len(SIZES)}")

    reduction_accuracy()

    if FAILS:
        print(f"\nFAILURES: {len(FAILS)}")
        for f in FAILS[:20]:
            print("  ", f)
        return 1
    print("\nAll kernels within their stated standard.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
