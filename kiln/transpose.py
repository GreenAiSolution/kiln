"""
kiln.transpose - transpose a float32 matrix, in registers.

The naive transpose is a scatter: read a row, write a column, one cache line
touched per element. This one loads a 4x4 block into four vector registers
and turns it inside out with eight shuffle instructions, so both the reads
and the writes stay contiguous:

    trn1/trn2 interleave the even and odd float32 lanes of two registers
    zip1/zip2 on .2d then swap the 64-bit halves

Four loads, eight shuffles, four stores, sixteen elements transposed.

Backpropagation needs this twice per layer, and both times on a matrix whose
size is small next to the multiply it feeds, so it never shows up in a
profile - but it has to exist, and it has to be right.

Pure standard library.
"""

from . import isa, jit
from .asm import Asm

X_SRC, X_DST, X_S, X_D = 0, 1, 2, 3
X_RCNT, X_CCNT = 16, 17
X_SSTEP, X_DSTEP = 4, 5


class TransposeError(Exception):
    pass


def pick_tile(R, C):
    """Largest square tile that divides both sides.

    Without tiling, finishing one strip of source rows writes to every row of
    the destination, so a 512x512 transpose touches 512 different pages
    before it comes back to the first one. Working a tile at a time keeps
    both the source block and the destination block resident, and on this
    machine that is worth about 2.5x.
    """
    # Measured on this machine: 512x512 peaks at tile 16 (4.9x numpy) and
    # 1024x1024 at tile 32 (5.9x); tile 64 is worse than both because the
    # destination block stops fitting. Tile 4 - no tiling at all - is 0.9x.
    order = (32, 16, 8, 4) if min(R, C) >= 512 else (16, 32, 8, 4)
    for t in order:
        if R % t == 0 and C % t == 0:
            return t
    return 4


def _emit_4x4(a, src, dst, C, R):
    """Load a 4x4 block from src, transpose it in registers, store to dst."""
    for i in range(4):
        a(isa.LDR_q(i, src, i * C * 4))
    a(isa.TRN1(4, 0, 1))
    a(isa.TRN2(5, 0, 1))
    a(isa.TRN1(6, 2, 3))
    a(isa.TRN2(7, 2, 3))
    a(isa.ZIP1_d(0, 4, 6))
    a(isa.ZIP1_d(1, 5, 7))
    a(isa.ZIP2_d(2, 4, 6))
    a(isa.ZIP2_d(3, 5, 7))
    for i in range(4):
        a(isa.STR_q(i, dst, i * R * 4))


def build(R, C, tile=None):
    """void f(const float *src, float *dst) for src[R][C] -> dst[C][R]."""
    if R % 4 or C % 4:
        raise TransposeError(f"transpose needs R%4==0 and C%4==0, got {R}x{C}")
    if 3 * C * 4 + 48 > 65520 or 3 * R * 4 + 48 > 65520:
        raise TransposeError("matrix too wide for immediate addressing")
    T = tile or pick_tile(R, C)

    # x2/x3 tile-row cursors, x4/x5 tile cursors, x6/x7 block-row cursors,
    # x14/x15 block cursors
    a = Asm(f"transpose_{R}x{C}_t{T}")
    a.mov_imm(8, 4 * C * 4)            # four source rows
    a.mov_imm(9, 4 * R * 4)            # four destination rows
    a.mov_imm(10, T * C * 4)           # one tile-row of source
    a.mov_imm(11, T * R * 4)           # one tile-row of destination
    a(isa.MOV_reg(2, X_SRC))
    a(isa.MOV_reg(3, X_DST))
    a.mov_imm(X_RCNT, R // T)

    a.label("tile_row")
    a(isa.MOV_reg(4, 2))
    a(isa.MOV_reg(5, 3))
    a.mov_imm(X_CCNT, C // T)

    a.label("tile_col")
    a(isa.MOV_reg(6, 4))
    a(isa.MOV_reg(7, 5))
    a.mov_imm(12, T // 4)

    a.label("block_row")
    a(isa.MOV_reg(14, 6))
    a(isa.MOV_reg(15, 7))
    a.mov_imm(13, T // 4)

    a.label("block_col")
    _emit_4x4(a, 14, 15, C, R)
    a(isa.ADD_imm(14, 14, 16))         # four columns along the source
    a(isa.ADD_reg(15, 15, 9))          # which is four rows down the result
    a(isa.SUBS_imm(13, 13, 1))
    a.bcond("ne", "block_col")

    a(isa.ADD_reg(6, 6, 8))
    a(isa.ADD_imm(7, 7, 16))
    a(isa.SUBS_imm(12, 12, 1))
    a.bcond("ne", "block_row")

    a(isa.ADD_imm(4, 4, T * 4))
    a(isa.ADD_reg(5, 5, 11))
    a(isa.SUBS_imm(X_CCNT, X_CCNT, 1))
    a.bcond("ne", "tile_col")

    a(isa.ADD_reg(2, 2, 10))
    a(isa.ADD_imm(3, 3, T * 4))
    a(isa.SUBS_imm(X_RCNT, X_RCNT, 1))
    a.bcond("ne", "tile_row")
    a(isa.RET())
    return a


class Transpose:
    def __init__(self, R, C, tile=None):
        self.R, self.C = R, C
        self.tile = tile or pick_tile(R, C)
        self.asm = build(R, C, tile)
        self.code = jit.load(self.asm)
        self.fn = self.code.fn(None, [jit.VOIDP, jit.VOIDP])

    def __call__(self, src, dst):
        self.fn(src.ptr, dst.ptr)


_cache = {}


def transpose(R, C, tile=None):
    t = _cache.get((R, C, tile))
    if t is None:
        t = _cache[(R, C, tile)] = Transpose(R, C, tile)
    return t
