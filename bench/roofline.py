"""
How fast can this core possibly go, and how close does KILN get?

A speedup number on its own is unfalsifiable - fast compared to what? So
before comparing against anything, KILN measures the two ceilings of the
machine it is running on, using kernels it emits itself:

  compute ceiling   a loop of nothing but independent fused multiply-adds,
                    no memory traffic at all. Whatever rate that hits is the
                    most this core's vector units can do.

  memory ceiling    a loop of nothing but loads, one accumulate each, over
                    an array far larger than any cache. Whatever rate that
                    hits is the most this core can pull from DRAM.

Every kernel then gets scored as a fraction of whichever ceiling binds it.
That is the roofline model, and it is the honest way to say "good".

Pure standard library, plus numpy for the comparison rows.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kiln import gemm, isa, jit                        # noqa: E402
from kiln.asm import Asm                               # noqa: E402
from kiln.runtime import bench_batched                 # noqa: E402

FMA_CHAINS = 24        # independent accumulators, to cover FMA latency
FMA_PER_ITER = 96      # FMAs in one unrolled loop body


def measure_fma_peak(iters=200000):
    """A loop of independent FMLAs on registers only. No loads, no stores,
    no dependency between consecutive instructions - so the only thing this
    can be limited by is how many vector FMAs the core retires per cycle."""
    a = Asm("fma_peak")
    for v in range(FMA_CHAINS):
        a(isa.MOVI_zero(v))
    a(isa.MOVI_zero(30))
    a(isa.MOVI_zero(31))
    a.mov_imm(16, iters)
    a.label("loop")
    for _ in range(FMA_PER_ITER // FMA_CHAINS):
        for v in range(FMA_CHAINS):
            a(isa.FMLA(v, 30, 31))
    a(isa.SUBS_imm(16, 16, 1))
    a.bcond("ne", "loop")
    a(isa.RET())

    code = jit.load(a)
    fn = code.fn(None, [])
    r = bench_batched(lambda: fn(), seconds=0.4)
    total_fma = iters * (FMA_PER_ITER // FMA_CHAINS) * FMA_CHAINS
    flops = total_fma * 4 * 2
    return {
        "gflops": flops / r["best"] / 1e9,
        "fma_per_cycle_at_3_2GHz": total_fma / r["best"] / 3.228e9,
        "seconds": r["best"],
        "insns": len(a),
    }


def measure_load_peak(n=1 << 24, unroll=8):
    """A loop that does nothing but stream an array through the vector unit.
    One accumulate per load, so the arithmetic can never be the limit."""
    a = Asm("load_peak")
    for v in range(unroll):
        a(isa.MOVI_zero(v))
    a(isa.LDR_x(4, 0, 0))
    nvec = n // 4
    blocks = nvec // unroll
    a.mov_imm(16, blocks)
    a.label("loop")
    for u in range(unroll):
        a(isa.LDR_q(16 + u, 4, u * 16))
    for u in range(unroll):
        a(isa.FADD(u, u, 16 + u))
    a(isa.ADD_imm(4, 4, unroll * 16))
    a(isa.SUBS_imm(16, 16, 1))
    a.bcond("ne", "loop")
    for u in range(1, unroll):
        a(isa.FADD(0, 0, u))
    a(isa.STR_q(0, 1, 0))
    a(isa.RET())

    code = jit.load(a)
    fn = code.fn(None, [jit.VOIDP, jit.VOIDP])
    src = jit.Buf(n, 1.0)
    dst = jit.Buf(4)
    table = (__import__("ctypes").c_void_p * 1)()
    table[0] = src.ptr
    import ctypes
    tp = ctypes.cast(table, ctypes.c_void_p)
    r = bench_batched(lambda: fn(tp, dst.ptr), seconds=0.4)
    return {
        "gbs": blocks * unroll * 16 / r["best"] / 1e9,
        "seconds": r["best"],
        "bytes": blocks * unroll * 16,
    }


def main():
    print("Machine ceilings, measured by KILN's own emitted code")
    print("=" * 74)
    fma = measure_fma_peak()
    print(f"vector FMA peak      : {fma['gflops']:8.1f} GFLOP/s  "
          f"({fma['fma_per_cycle_at_3_2GHz']:.2f} FMA/cycle at 3.228 GHz)")
    mem = measure_load_peak()
    print(f"streaming load peak  : {mem['gbs']:8.1f} GB/s      "
          f"(single core, {mem['bytes'] / 1e6:.0f} MB per pass)")

    print()
    print("KILN matmul against the compute ceiling")
    print(f"  {'shape':>16} {'GFLOP/s':>9} {'% of peak':>10} "
          f"{'flops/insn':>11} {'code':>7}")
    rows = []
    for s in (64, 128, 256, 512, 1024, 2048):
        try:
            g = gemm.gemm(s, s, s)
        except gemm.GemmError as e:
            print(f"  {s}^3 skipped: {e}")
            continue
        A, B, C = jit.Buf(s * s, 0.5), jit.Buf(s * s, 0.25), jit.Buf(s * s)
        r = bench_batched(lambda: g(A, B, C), seconds=0.3)
        gf = g.flops / r["best"] / 1e9
        rep = g.report()
        print(f"  {s}x{s}x{s:<8} {gf:>9.1f} {100 * gf / fma['gflops']:>9.1f}% "
              f"{rep['flops_per_insn']:>11.2f} {rep['code_bytes']:>6}B")
        rows.append((s, gf))

    print()
    print("What the ceiling means")
    print(f"  This core can retire about "
          f"{fma['fma_per_cycle_at_3_2GHz']:.1f} 4-wide FMAs per cycle.")
    best = max(r[1] for r in rows) if rows else 0
    print(f"  KILN's matmul reaches {100 * best / fma['gflops']:.0f}% of that "
          f"with no packing pass and no assembler.")
    print()
    print("  Apple's Accelerate beats this by roughly an order of magnitude,")
    print("  and not by writing better NEON: it dispatches to AMX, an on-die")
    print("  matrix coprocessor whose instruction encoding Apple does not")
    print("  publish. There is no sequence of documented ARM instructions that")
    print("  reaches it. That is the real ceiling on 'from scratch' here, and")
    print("  it is worth naming rather than hiding.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
