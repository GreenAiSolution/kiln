"""
kiln.gemm - matrix multiply, emitted as ARM64 NEON machine code.

C[M,N] = A[M,K] @ B[K,N], all float32, all row-major.

The micro-kernel holds an 8x8 block of C in 16 vector registers and never
writes it to memory until the whole K dimension is consumed. Each step of
the inner loop:

    load 2 vectors of B  (8 columns of one k row)
    issue 16 fused multiply-adds against a broadcast lane of A

so sixteen FMAs ride on two loads. A is read four k-values at a time and
each of those four values is broadcast straight out of the register with
FMLA's lane-indexed form, which is why no packing pass is needed: both
operands are read in the layout they already have.

All three loop trip counts are baked in as constants, because a JIT knows
the shape. Register allocation is fixed by hand here rather than searched -
at this register pressure there is exactly one sensible assignment.

Pure standard library.
"""

from . import isa, jit
from .asm import Asm

MR, NR, KB = 8, 8, 4          # rows, columns, k-values per inner step

# general-purpose register assignment
X_A, X_B, X_C = 0, 1, 2
X_APAN, X_CPAN = 3, 4         # A and C row-panel cursors
X_BCOL, X_CTILE = 5, 6        # B column and C tile cursors
X_AK, X_BK = 7, 8             # k cursors
X_KCNT, X_BKSTRIDE = 9, 10
X_ASTEP, X_CSTEP = 11, 12
X_MCNT, X_NCNT = 16, 17

# vector registers: 16 accumulators, 8 A rows, 2 B columns
ACC = [[i * 2, i * 2 + 1] for i in range(MR)]     # v0..v15
AREG = [16 + i for i in range(MR)]                # v16..v23
BREG = [24, 25]                                   # v24, v25

IMM_LIMIT = 65520             # largest scaled offset an LDR Q can encode


class GemmError(Exception):
    pass


def check_shape(M, N, K, lda, ldb, ldc):
    if M % MR or N % NR or K % KB:
        raise GemmError(f"gemm needs M%{MR}==0, N%{NR}==0, K%{KB}==0; "
                        f"got {M}x{N}x{K}")
    if (MR - 1) * lda * 4 > IMM_LIMIT:
        raise GemmError(f"lda={lda} too large for immediate addressing")
    if (MR - 1) * ldc * 4 + 16 > IMM_LIMIT:
        raise GemmError(f"ldc={ldc} too large for immediate addressing")
    if (KB - 1) * ldb * 4 + 16 > IMM_LIMIT:
        raise GemmError(f"ldb={ldb} too large for immediate addressing")


def build(M, N, K, accumulate=False, lda=None, ldb=None, ldc=None):
    """Emit the whole triple loop as one function: void f(A, B, C).

    lda/ldb/ldc are the row strides of the full matrices, which differ from
    M/N/K when this kernel is being used on a cache-sized block of a larger
    problem.
    """
    lda = K if lda is None else lda
    ldb = N if ldb is None else ldb
    ldc = N if ldc is None else ldc
    check_shape(M, N, K, lda, ldb, ldc)
    a = Asm(f"gemm_{M}x{N}x{K}")

    a(isa.MOV_reg(X_APAN, X_A))
    a(isa.MOV_reg(X_CPAN, X_C))
    a.mov_imm(X_BKSTRIDE, KB * ldb * 4)   # B advances this far per inner step
    a.mov_imm(X_ASTEP, MR * lda * 4)      # A advances this far per m-tile
    a.mov_imm(X_CSTEP, MR * ldc * 4)
    a.mov_imm(X_MCNT, M // MR)

    a.label("m_loop")
    a(isa.MOV_reg(X_BCOL, X_B))
    a(isa.MOV_reg(X_CTILE, X_CPAN))
    a.mov_imm(X_NCNT, N // NR)

    a.label("n_loop")
    if accumulate:
        for i in range(MR):
            a(isa.LDR_q(ACC[i][0], X_CTILE, i * ldc * 4))
            a(isa.LDR_q(ACC[i][1], X_CTILE, i * ldc * 4 + 16))
    else:
        for i in range(MR):
            a(isa.MOVI_zero(ACC[i][0]))
            a(isa.MOVI_zero(ACC[i][1]))

    a(isa.MOV_reg(X_AK, X_APAN))
    a(isa.MOV_reg(X_BK, X_BCOL))
    a.mov_imm(X_KCNT, K // KB)

    a.label("k_loop")
    # four k-values of eight A rows, one vector each
    for i in range(MR):
        a(isa.LDR_q(AREG[i], X_AK, i * lda * 4))
    for kk in range(KB):
        a(isa.LDR_q(BREG[0], X_BK, kk * ldb * 4))
        a(isa.LDR_q(BREG[1], X_BK, kk * ldb * 4 + 16))
        for i in range(MR):
            a(isa.FMLA_lane(ACC[i][0], BREG[0], AREG[i], kk))
            a(isa.FMLA_lane(ACC[i][1], BREG[1], AREG[i], kk))
    a(isa.ADD_imm(X_AK, X_AK, KB * 4))
    a(isa.ADD_reg(X_BK, X_BK, X_BKSTRIDE))
    a(isa.SUBS_imm(X_KCNT, X_KCNT, 1))
    a.bcond("ne", "k_loop")

    for i in range(MR):
        a(isa.STR_q(ACC[i][0], X_CTILE, i * ldc * 4))
        a(isa.STR_q(ACC[i][1], X_CTILE, i * ldc * 4 + 16))

    a(isa.ADD_imm(X_BCOL, X_BCOL, NR * 4))
    a(isa.ADD_imm(X_CTILE, X_CTILE, NR * 4))
    a(isa.SUBS_imm(X_NCNT, X_NCNT, 1))
    a.bcond("ne", "n_loop")

    a(isa.ADD_reg(X_APAN, X_APAN, X_ASTEP))
    a(isa.ADD_reg(X_CPAN, X_CPAN, X_CSTEP))
    a(isa.SUBS_imm(X_MCNT, X_MCNT, 1))
    a.bcond("ne", "m_loop")

    a(isa.RET())
    return a


class Gemm:
    """A compiled multiply for one specific shape."""

    def __init__(self, M, N, K, accumulate=False, lda=None, ldb=None, ldc=None):
        self.M, self.N, self.K = M, N, K
        self.lda = K if lda is None else lda
        self.ldb = N if ldb is None else ldb
        self.ldc = N if ldc is None else ldc
        self.accumulate = accumulate
        self.asm = build(M, N, K, accumulate, lda, ldb, ldc)
        self.code = jit.load(self.asm)
        self.fn = self.code.fn(None, [jit.VOIDP, jit.VOIDP, jit.VOIDP])
        self.flops = 2 * M * N * K

    def __call__(self, A, B, C):
        self.fn(A.ptr, B.ptr, C.ptr)

    def inner_insns(self):
        """Instructions in the innermost loop, for the roofline maths."""
        return MR + KB * (2 + 2 * MR) + 4

    def flops_per_inner(self):
        return KB * MR * 2 * 4 * 2

    def report(self):
        return {
            "shape": (self.M, self.N, self.K),
            "insns": len(self.asm),
            "code_bytes": self.code.nbytes,
            "inner_insns": self.inner_insns(),
            "inner_flops": self.flops_per_inner(),
            "flops_per_insn": self.flops_per_inner() / self.inner_insns(),
        }


_cache = {}


def gemm(M, N, K, accumulate=False, lda=None, ldb=None, ldc=None):
    key = (M, N, K, accumulate, lda, ldb, ldc)
    g = _cache.get(key)
    if g is None:
        g = _cache[key] = Gemm(M, N, K, accumulate, lda, ldb, ldc)
    return g


# ------------------------------------------------------------ cache blocking

NC_DEFAULT = 256          # columns of B held resident
KC_DEFAULT = 256          # depth of the B block: NC*KC*4 bytes must fit L2


class BlockedGemm:
    """Multiply large matrices by walking cache-sized blocks of B.

    Without this, a big multiply re-streams a column strip of B from DRAM for
    every row panel of A, and at 2048x2048 that collapses to a fifth of peak.
    Holding a KC x NC block of B resident and sweeping all of A past it turns
    the traffic back into something the cache can absorb. The block sizes are
    the only real tuning knobs, and bench/roofline.py measures them rather
    than assuming.
    """

    def __init__(self, M, N, K, nc=NC_DEFAULT, kc=KC_DEFAULT):
        self.M, self.N, self.K = M, N, K
        self.nc = min(nc - (nc % NR), N)
        self.kc = min(kc - (kc % KB), K)
        self.flops = 2 * M * N * K
        self.steps = []
        for n0 in range(0, N, self.nc):
            nw = min(self.nc, N - n0)
            for k0 in range(0, K, self.kc):
                kw = min(self.kc, K - k0)
                g = gemm(M, nw, kw, accumulate=(k0 > 0),
                         lda=K, ldb=N, ldc=N)
                self.steps.append((g, k0 * 4, (k0 * N + n0) * 4, n0 * 4))

    def __call__(self, A, B, C):
        ap, bp, cp = A.ptr, B.ptr, C.ptr
        for g, aoff, boff, coff in self.steps:
            g.fn(ap + aoff, bp + boff, cp + coff)

    def report(self):
        return {"blocks": len(self.steps), "nc": self.nc, "kc": self.kc,
                "code_bytes": sum(g.code.nbytes for g, _, _, _ in self.steps)}


def blocked(M, N, K, nc=NC_DEFAULT, kc=KC_DEFAULT):
    key = ("blocked", M, N, K, nc, kc)
    g = _cache.get(key)
    if g is None:
        g = _cache[key] = BlockedGemm(M, N, K, nc, kc)
    return g


# ------------------------------------------------------- padded convenience

def pad_to(v, mult):
    return ((v + mult - 1) // mult) * mult


def matmul(A, B, M, N, K, C=None):
    """Multiply with automatic padding for awkward shapes.

    Padding costs a copy, which is O(M*K + K*N) against O(M*N*K) of real
    work, so it disappears at any interesting size - but it is real and the
    benchmark reports padded and unpadded shapes separately.
    """
    Mp, Np, Kp = pad_to(M, MR), pad_to(N, NR), pad_to(K, KB)
    if (Mp, Np, Kp) == (M, N, K):
        C = C or jit.Buf(M * N)
        gemm(M, N, K)(A, B, C)
        return C
    Ap, Bp = jit.Buf(Mp * Kp), jit.Buf(Kp * Np)
    for i in range(M):
        Ap[i * Kp:i * Kp + K] = [A[i * K + j] for j in range(K)]
    for i in range(K):
        Bp[i * Np:i * Np + N] = [B[i * N + j] for j in range(N)]
    Cp = jit.Buf(Mp * Np)
    gemm(Mp, Np, Kp)(Ap, Bp, Cp)
    C = C or jit.Buf(M * N)
    for i in range(M):
        C[i * N:i * N + N] = [Cp[i * Np + j] for j in range(N)]
    return C
