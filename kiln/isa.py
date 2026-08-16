"""
kiln.isa - AArch64 (ARM64) instruction encoder.

Every function here returns an Insn: a 32-bit machine word AND the assembly
text that word is supposed to mean. Carrying both lets tests/verify_isa.py
hand the text to Apple's own assembler and compare byte-for-byte. Nothing in
this file is trusted on my say-so; it is all checked against clang.

Pure standard library. No dependencies.
"""

from collections import namedtuple

Insn = namedtuple("Insn", "w text")

# ---------------------------------------------------------------- registers

SP = 31        # in load/store base position, 31 means SP
XZR = 31       # in ALU operand position, 31 means the zero register
LR = 30
FP = 29


def _x(n):
    if n == 31:
        return "xzr"
    return f"x{n}"


def _xsp(n):
    if n == 31:
        return "sp"
    return f"x{n}"


def _w(n):
    if n == 31:
        return "wzr"
    return f"w{n}"


def _v(n):
    return f"v{n}.4s"


def _chk(v, lo, hi, what):
    if not (lo <= v <= hi):
        raise ValueError(f"{what} out of range: {v} not in [{lo},{hi}]")
    return v


# ------------------------------------------------------------ control flow

def RET():
    return Insn(0xD65F03C0, "ret")


def B(imm26):
    """Unconditional branch. imm26 is in instructions, signed."""
    _chk(imm26, -(1 << 25), (1 << 25) - 1, "b offset")
    return Insn(0x14000000 | (imm26 & 0x03FFFFFF), f"b #{imm26 * 4}")


_CONDS = {
    "eq": 0, "ne": 1, "hs": 2, "lo": 3, "mi": 4, "pl": 5, "vs": 6, "vc": 7,
    "hi": 8, "ls": 9, "ge": 10, "lt": 11, "gt": 12, "le": 13, "al": 14,
}


def BCOND(cond, imm19):
    """Conditional branch. imm19 is in instructions, signed."""
    c = _CONDS[cond]
    _chk(imm19, -(1 << 18), (1 << 18) - 1, "b.cond offset")
    return Insn(0x54000000 | ((imm19 & 0x7FFFF) << 5) | c,
                f"b.{cond} #{imm19 * 4}")


def CBNZ(rt, imm19):
    _chk(imm19, -(1 << 18), (1 << 18) - 1, "cbnz offset")
    return Insn(0xB5000000 | ((imm19 & 0x7FFFF) << 5) | rt,
                f"cbnz {_x(rt)}, #{imm19 * 4}")


def CBZ(rt, imm19):
    _chk(imm19, -(1 << 18), (1 << 18) - 1, "cbz offset")
    return Insn(0xB4000000 | ((imm19 & 0x7FFFF) << 5) | rt,
                f"cbz {_x(rt)}, #{imm19 * 4}")


# --------------------------------------------------------- integer ALU (64)

def ADD_imm(rd, rn, imm12):
    _chk(imm12, 0, 4095, "add imm")
    return Insn(0x91000000 | (imm12 << 10) | (rn << 5) | rd,
                f"add {_xsp(rd)}, {_xsp(rn)}, #{imm12}")


def SUB_imm(rd, rn, imm12):
    _chk(imm12, 0, 4095, "sub imm")
    return Insn(0xD1000000 | (imm12 << 10) | (rn << 5) | rd,
                f"sub {_xsp(rd)}, {_xsp(rn)}, #{imm12}")


def SUBS_imm(rd, rn, imm12):
    _chk(imm12, 0, 4095, "subs imm")
    return Insn(0xF1000000 | (imm12 << 10) | (rn << 5) | rd,
                f"subs {_x(rd)}, {_xsp(rn)}, #{imm12}")


def ADD_reg(rd, rn, rm):
    return Insn(0x8B000000 | (rm << 16) | (rn << 5) | rd,
                f"add {_x(rd)}, {_x(rn)}, {_x(rm)}")


def SUB_reg(rd, rn, rm):
    return Insn(0xCB000000 | (rm << 16) | (rn << 5) | rd,
                f"sub {_x(rd)}, {_x(rn)}, {_x(rm)}")


def ADD_reg_lsl(rd, rn, rm, sh):
    _chk(sh, 0, 63, "lsl amount")
    return Insn(0x8B000000 | (rm << 16) | (sh << 10) | (rn << 5) | rd,
                f"add {_x(rd)}, {_x(rn)}, {_x(rm)}, lsl #{sh}")


def MOV_reg(rd, rm):
    return Insn(0xAA0003E0 | (rm << 16) | rd, f"mov {_x(rd)}, {_x(rm)}")


def MOVZ(rd, imm16, shift=0):
    _chk(imm16, 0, 0xFFFF, "movz imm")
    hw = shift // 16
    _chk(hw, 0, 3, "movz shift")
    txt = f"movz {_x(rd)}, #{imm16}"
    if shift:
        txt += f", lsl #{shift}"
    return Insn(0xD2800000 | (hw << 21) | (imm16 << 5) | rd, txt)


def MOVK(rd, imm16, shift=0):
    _chk(imm16, 0, 0xFFFF, "movk imm")
    hw = shift // 16
    _chk(hw, 0, 3, "movk shift")
    txt = f"movk {_x(rd)}, #{imm16}"
    if shift:
        txt += f", lsl #{shift}"
    return Insn(0xF2800000 | (hw << 21) | (imm16 << 5) | rd, txt)


def LSL_imm(rd, rn, sh):
    """LSL Xd, Xn, #sh  (alias of UBFM)."""
    _chk(sh, 0, 63, "lsl imm")
    immr = (64 - sh) % 64
    imms = 63 - sh
    return Insn(0xD3400000 | (immr << 16) | (imms << 10) | (rn << 5) | rd,
                f"lsl {_x(rd)}, {_x(rn)}, #{sh}")


def MADD(rd, rn, rm, ra):
    return Insn(0x9B000000 | (rm << 16) | (ra << 10) | (rn << 5) | rd,
                f"madd {_x(rd)}, {_x(rn)}, {_x(rm)}, {_x(ra)}")


def MUL(rd, rn, rm):
    return Insn(0x9B000000 | (rm << 16) | (31 << 10) | (rn << 5) | rd,
                f"mul {_x(rd)}, {_x(rn)}, {_x(rm)}")


def CMP_imm(rn, imm12):
    _chk(imm12, 0, 4095, "cmp imm")
    return Insn(0xF100001F | (imm12 << 10) | (rn << 5),
                f"cmp {_xsp(rn)}, #{imm12}")


def CMP_reg(rn, rm):
    return Insn(0xEB00001F | (rm << 16) | (rn << 5),
                f"cmp {_x(rn)}, {_x(rm)}")


# ----------------------------------------------------------- memory (int)

def STP_pre(rt, rt2, rn, off):
    """STP Xt, Xt2, [Xn, #off]!"""
    if off % 8:
        raise ValueError("stp offset must be a multiple of 8")
    imm7 = off // 8
    _chk(imm7, -64, 63, "stp imm7")
    return Insn(0xA9800000 | ((imm7 & 0x7F) << 15) | (rt2 << 10) | (rn << 5) | rt,
                f"stp {_x(rt)}, {_x(rt2)}, [{_xsp(rn)}, #{off}]!")


def LDP_post(rt, rt2, rn, off):
    """LDP Xt, Xt2, [Xn], #off"""
    if off % 8:
        raise ValueError("ldp offset must be a multiple of 8")
    imm7 = off // 8
    _chk(imm7, -64, 63, "ldp imm7")
    return Insn(0xA8C00000 | ((imm7 & 0x7F) << 15) | (rt2 << 10) | (rn << 5) | rt,
                f"ldp {_x(rt)}, {_x(rt2)}, [{_xsp(rn)}], #{off}")


def LDR_x(rt, rn, off=0):
    """LDR Xt, [Xn, #off]  (unsigned scaled offset)"""
    if off % 8:
        raise ValueError("ldr x offset must be a multiple of 8")
    imm12 = off // 8
    _chk(imm12, 0, 4095, "ldr imm12")
    return Insn(0xF9400000 | (imm12 << 10) | (rn << 5) | rt,
                f"ldr {_x(rt)}, [{_xsp(rn)}, #{off}]")


def STR_x(rt, rn, off=0):
    if off % 8:
        raise ValueError("str x offset must be a multiple of 8")
    imm12 = off // 8
    _chk(imm12, 0, 4095, "str imm12")
    return Insn(0xF9000000 | (imm12 << 10) | (rn << 5) | rt,
                f"str {_x(rt)}, [{_xsp(rn)}, #{off}]")


# -------------------------------------------------------- memory (vector)

def LDR_q(rt, rn, off=0):
    """LDR Qt, [Xn, #off] - load 16 bytes (four float32 lanes)."""
    if off % 16:
        raise ValueError("ldr q offset must be a multiple of 16")
    imm12 = off // 16
    _chk(imm12, 0, 4095, "ldr q imm12")
    return Insn(0x3DC00000 | (imm12 << 10) | (rn << 5) | rt,
                f"ldr q{rt}, [{_xsp(rn)}, #{off}]")


def STR_q(rt, rn, off=0):
    if off % 16:
        raise ValueError("str q offset must be a multiple of 16")
    imm12 = off // 16
    _chk(imm12, 0, 4095, "str q imm12")
    return Insn(0x3D800000 | (imm12 << 10) | (rn << 5) | rt,
                f"str q{rt}, [{_xsp(rn)}, #{off}]")


def LDR_s(rt, rn, off=0):
    """LDR St, [Xn, #off] - load one float32."""
    if off % 4:
        raise ValueError("ldr s offset must be a multiple of 4")
    imm12 = off // 4
    _chk(imm12, 0, 4095, "ldr s imm12")
    return Insn(0xBD400000 | (imm12 << 10) | (rn << 5) | rt,
                f"ldr s{rt}, [{_xsp(rn)}, #{off}]")


def STR_s(rt, rn, off=0):
    if off % 4:
        raise ValueError("str s offset must be a multiple of 4")
    imm12 = off // 4
    _chk(imm12, 0, 4095, "str s imm12")
    return Insn(0xBD000000 | (imm12 << 10) | (rn << 5) | rt,
                f"str s{rt}, [{_xsp(rn)}, #{off}]")


def LDR_q_reg(rt, rn, rm):
    """LDR Qt, [Xn, Xm, lsl #4] - indexed by vector count."""
    return Insn(0x3CE07800 | (rm << 16) | (rn << 5) | rt,
                f"ldr q{rt}, [{_xsp(rn)}, {_x(rm)}, lsl #4]")


def PRFM(rn, off=0):
    """PRFM PLDL1KEEP, [Xn, #off]"""
    if off % 8:
        raise ValueError("prfm offset must be a multiple of 8")
    imm12 = off // 8
    _chk(imm12, 0, 4095, "prfm imm12")
    return Insn(0xF9800000 | (imm12 << 10) | (rn << 5) | 0,
                f"prfm pldl1keep, [{_xsp(rn)}, #{off}]")


# ------------------------------------------------------- NEON float32 x4
# Arrangement is always .4s here: Q=1, sz=0 (or sz=1 where the bit selects
# the sub-opcode rather than the element size).

def FMLA(rd, rn, rm):
    """Vd.4s += Vn.4s * Vm.4s  - the fused multiply-add this whole thing is built on."""
    return Insn(0x4E20CC00 | (rm << 16) | (rn << 5) | rd,
                f"fmla {_v(rd)}, {_v(rn)}, {_v(rm)}")


def FMLS(rd, rn, rm):
    return Insn(0x4EA0CC00 | (rm << 16) | (rn << 5) | rd,
                f"fmls {_v(rd)}, {_v(rn)}, {_v(rm)}")


def FMLA_lane(rd, rn, rm, idx):
    """Vd.4s += Vn.4s * Vm.s[idx] - broadcast one lane, the matmul workhorse."""
    _chk(idx, 0, 3, "fmla lane")
    H = (idx >> 1) & 1
    L = idx & 1
    M = (rm >> 4) & 1
    return Insn(0x4F801000 | (L << 21) | (M << 20) | ((rm & 15) << 16)
                | (H << 11) | (rn << 5) | rd,
                f"fmla {_v(rd)}, {_v(rn)}, v{rm}.s[{idx}]")


def FMUL(rd, rn, rm):
    return Insn(0x6E20DC00 | (rm << 16) | (rn << 5) | rd,
                f"fmul {_v(rd)}, {_v(rn)}, {_v(rm)}")


def FADD(rd, rn, rm):
    return Insn(0x4E20D400 | (rm << 16) | (rn << 5) | rd,
                f"fadd {_v(rd)}, {_v(rn)}, {_v(rm)}")


def FSUB(rd, rn, rm):
    return Insn(0x4EA0D400 | (rm << 16) | (rn << 5) | rd,
                f"fsub {_v(rd)}, {_v(rn)}, {_v(rm)}")


def FDIV(rd, rn, rm):
    return Insn(0x6E20FC00 | (rm << 16) | (rn << 5) | rd,
                f"fdiv {_v(rd)}, {_v(rn)}, {_v(rm)}")


def FMAX(rd, rn, rm):
    return Insn(0x4E20F400 | (rm << 16) | (rn << 5) | rd,
                f"fmax {_v(rd)}, {_v(rn)}, {_v(rm)}")


def FMIN(rd, rn, rm):
    return Insn(0x4EA0F400 | (rm << 16) | (rn << 5) | rd,
                f"fmin {_v(rd)}, {_v(rn)}, {_v(rm)}")


def FNEG(rd, rn):
    return Insn(0x6EA0F800 | (rn << 5) | rd, f"fneg {_v(rd)}, {_v(rn)}")


def FABS(rd, rn):
    return Insn(0x4EA0F800 | (rn << 5) | rd, f"fabs {_v(rd)}, {_v(rn)}")


def FSQRT(rd, rn):
    return Insn(0x6EA1F800 | (rn << 5) | rd, f"fsqrt {_v(rd)}, {_v(rn)}")


def FRECPE(rd, rn):
    return Insn(0x4EA1D800 | (rn << 5) | rd, f"frecpe {_v(rd)}, {_v(rn)}")


def FRECPS(rd, rn, rm):
    return Insn(0x4E20FC00 | (rm << 16) | (rn << 5) | rd,
                f"frecps {_v(rd)}, {_v(rn)}, {_v(rm)}")


def FRSQRTE(rd, rn):
    return Insn(0x6EA1D800 | (rn << 5) | rd, f"frsqrte {_v(rd)}, {_v(rn)}")


def FRSQRTS(rd, rn, rm):
    return Insn(0x4EA0FC00 | (rm << 16) | (rn << 5) | rd,
                f"frsqrts {_v(rd)}, {_v(rn)}, {_v(rm)}")


def FRINTN(rd, rn):
    """Round to nearest, ties to even."""
    return Insn(0x4E218800 | (rn << 5) | rd, f"frintn {_v(rd)}, {_v(rn)}")


def FCMGT(rd, rn, rm):
    return Insn(0x6EA0E400 | (rm << 16) | (rn << 5) | rd,
                f"fcmgt {_v(rd)}, {_v(rn)}, {_v(rm)}")


def FCMGE(rd, rn, rm):
    return Insn(0x6E20E400 | (rm << 16) | (rn << 5) | rd,
                f"fcmge {_v(rd)}, {_v(rn)}, {_v(rm)}")


def FCVTZS(rd, rn):
    """float32 -> int32, truncate toward zero."""
    return Insn(0x4EA1B800 | (rn << 5) | rd, f"fcvtzs {_v(rd)}, {_v(rn)}")


def SCVTF(rd, rn):
    """int32 -> float32."""
    return Insn(0x4E21D800 | (rn << 5) | rd, f"scvtf {_v(rd)}, {_v(rn)}")


# integer lanes (used to build exp: exponent surgery)

def VADD(rd, rn, rm):
    return Insn(0x4EA08400 | (rm << 16) | (rn << 5) | rd,
                f"add {_v(rd)}, {_v(rn)}, {_v(rm)}")


def VSUB(rd, rn, rm):
    return Insn(0x6EA08400 | (rm << 16) | (rn << 5) | rd,
                f"sub {_v(rd)}, {_v(rn)}, {_v(rm)}")


def VSHL(rd, rn, sh):
    """SHL Vd.4s, Vn.4s, #sh"""
    _chk(sh, 0, 31, "shl amount")
    immhb = 32 + sh
    return Insn(0x4F005400 | (immhb << 16) | (rn << 5) | rd,
                f"shl {_v(rd)}, {_v(rn)}, #{sh}")


def SMAX(rd, rn, rm):
    return Insn(0x4EA06400 | (rm << 16) | (rn << 5) | rd,
                f"smax {_v(rd)}, {_v(rn)}, {_v(rm)}")


def SMIN(rd, rn, rm):
    return Insn(0x4EA06C00 | (rm << 16) | (rn << 5) | rd,
                f"smin {_v(rd)}, {_v(rn)}, {_v(rm)}")


def AND_v(rd, rn, rm):
    return Insn(0x4E201C00 | (rm << 16) | (rn << 5) | rd,
                f"and v{rd}.16b, v{rn}.16b, v{rm}.16b")


def ORR_v(rd, rn, rm):
    return Insn(0x4EA01C00 | (rm << 16) | (rn << 5) | rd,
                f"orr v{rd}.16b, v{rn}.16b, v{rm}.16b")


def EOR_v(rd, rn, rm):
    return Insn(0x6E201C00 | (rm << 16) | (rn << 5) | rd,
                f"eor v{rd}.16b, v{rn}.16b, v{rm}.16b")


def BSL(rd, rn, rm):
    """Bitwise select: rd = (rn & rd) | (rm & ~rd)."""
    return Insn(0x6E601C00 | (rm << 16) | (rn << 5) | rd,
                f"bsl v{rd}.16b, v{rn}.16b, v{rm}.16b")


def MOVI_zero(rd):
    """Zero a whole vector register."""
    return Insn(0x6F00E400 | rd, f"movi v{rd}.2d, #0000000000000000")


def DUP_from_w(rd, rn):
    """Broadcast a general register's low 32 bits to all four lanes."""
    return Insn(0x4E040C00 | (rn << 5) | rd, f"dup {_v(rd)}, {_w(rn)}")


def DUP_lane(rd, rn, idx):
    _chk(idx, 0, 3, "dup lane")
    imm5 = (idx << 3) | 4
    return Insn(0x4E000400 | (imm5 << 16) | (rn << 5) | rd,
                f"dup {_v(rd)}, v{rn}.s[{idx}]")


def FADDP(rd, rn, rm):
    """Pairwise add across two vectors."""
    return Insn(0x6E20D400 | (rm << 16) | (rn << 5) | rd,
                f"faddp {_v(rd)}, {_v(rn)}, {_v(rm)}")


def FADDP_s(rd, rn):
    """FADDP Sd, Vn.2s - final step of a horizontal sum."""
    return Insn(0x7E30D800 | (rn << 5) | rd, f"faddp s{rd}, v{rn}.2s")


def FMAXV(rd, rn):
    """FMAXV Sd, Vn.4s - horizontal max across four lanes."""
    return Insn(0x6E30F800 | (rn << 5) | rd, f"fmaxv s{rd}, {_v(rn)}")


def TRN1(rd, rn, rm):
    """Interleave even-indexed float32 lanes. Half of a 4x4 transpose."""
    return Insn(0x4E802800 | (rm << 16) | (rn << 5) | rd,
                f"trn1 {_v(rd)}, {_v(rn)}, {_v(rm)}")


def TRN2(rd, rn, rm):
    return Insn(0x4E806800 | (rm << 16) | (rn << 5) | rd,
                f"trn2 {_v(rd)}, {_v(rn)}, {_v(rm)}")


def ZIP1_d(rd, rn, rm):
    """Take the low 64-bit half of each source. The other half of a transpose."""
    return Insn(0x4EC03800 | (rm << 16) | (rn << 5) | rd,
                f"zip1 v{rd}.2d, v{rn}.2d, v{rm}.2d")


def ZIP2_d(rd, rn, rm):
    return Insn(0x4EC07800 | (rm << 16) | (rn << 5) | rd,
                f"zip2 v{rd}.2d, v{rn}.2d, v{rm}.2d")


def MOV_v(rd, rn):
    """MOV Vd.16b, Vn.16b (alias of ORR)."""
    return Insn(0x4EA01C00 | (rn << 16) | (rn << 5) | rd,
                f"mov v{rd}.16b, v{rn}.16b")


# ------------------------------------------------------------ scalar float

def FMOV_s_from_w(rd, rn):
    return Insn(0x1E270000 | (rn << 5) | rd, f"fmov s{rd}, {_w(rn)}")


def FMOV_w_from_s(rd, rn):
    return Insn(0x1E260000 | (rn << 5) | rd, f"fmov {_w(rd)}, s{rn}")


def FADD_s(rd, rn, rm):
    return Insn(0x1E202800 | (rm << 16) | (rn << 5) | rd,
                f"fadd s{rd}, s{rn}, s{rm}")


def FMUL_s(rd, rn, rm):
    return Insn(0x1E200800 | (rm << 16) | (rn << 5) | rd,
                f"fmul s{rd}, s{rn}, s{rm}")


def FMADD_s(rd, rn, rm, ra):
    return Insn(0x1F000000 | (rm << 16) | (ra << 10) | (rn << 5) | rd,
                f"fmadd s{rd}, s{rn}, s{rm}, s{ra}")


def FMAX_s(rd, rn, rm):
    return Insn(0x1E204800 | (rm << 16) | (rn << 5) | rd,
                f"fmax s{rd}, s{rn}, s{rm}")


def FDIV_s(rd, rn, rm):
    return Insn(0x1E201800 | (rm << 16) | (rn << 5) | rd,
                f"fdiv s{rd}, s{rn}, s{rm}")


def FSUB_s(rd, rn, rm):
    return Insn(0x1E203800 | (rm << 16) | (rn << 5) | rd,
                f"fsub s{rd}, s{rn}, s{rm}")


def FSQRT_s(rd, rn):
    return Insn(0x1E21C000 | (rn << 5) | rd, f"fsqrt s{rd}, s{rn}")


# Every encoder in this module, for the differential test to enumerate.
ENCODERS = {
    name: obj for name, obj in list(globals().items())
    if callable(obj) and not name.startswith("_") and name.isupper()
       or (callable(obj) and not name.startswith("_")
           and name[0].isupper() and name not in ("Insn",))
}
