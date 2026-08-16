"""
Differential test: KILN's encoder vs Apple's assembler.

For every instruction KILN can emit, we take the assembly text the encoder
claims it means, hand that text to clang, and compare the 4 bytes clang
produces against the 4 bytes we produced. A single mismatched bit fails.

This is the reason to believe anything else in the project.
"""

import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kiln import isa  # noqa: E402


def cases():
    """(insn, label) for a spread of operands over every encoder."""
    out = []

    def add(i, tag):
        out.append((i, tag))

    # control flow
    add(isa.RET(), "ret")
    for off in (1, -1, 4, -64, 1000, -1000):
        add(isa.B(off), "b")
        add(isa.BCOND("ne", off), "b.ne")
        add(isa.BCOND("lt", off), "b.lt")
        add(isa.BCOND("ge", off), "b.ge")
        add(isa.BCOND("eq", off), "b.eq")
        add(isa.CBNZ(3, off), "cbnz")
        add(isa.CBZ(7, off), "cbz")

    # integer alu
    for rd, rn, rm in ((0, 1, 2), (9, 10, 11), (28, 29, 30), (5, 31, 0)):
        add(isa.ADD_reg(rd, rn, rm), "add reg")
        add(isa.SUB_reg(rd, rn, rm), "sub reg")
        add(isa.MOV_reg(rd, rm), "mov reg")
        add(isa.MUL(rd, rn, rm), "mul")
        add(isa.MADD(rd, rn, rm, 4), "madd")
        add(isa.CMP_reg(rn, rm), "cmp reg")
        for sh in (0, 2, 4, 16):
            add(isa.ADD_reg_lsl(rd, rn, rm, sh), "add lsl")
        for sh in (0, 1, 3, 8, 31, 63):
            add(isa.LSL_imm(rd, rn, sh), "lsl imm")
    for imm in (0, 1, 8, 4095, 16):
        add(isa.ADD_imm(1, 2, imm), "add imm")
        add(isa.SUB_imm(1, 2, imm), "sub imm")
        add(isa.SUBS_imm(1, 2, imm), "subs imm")
        add(isa.CMP_imm(3, imm), "cmp imm")
        add(isa.ADD_imm(31, 31, imm), "add sp")
    for imm in (0, 1, 65535, 4321):
        for sh in (0, 16, 32, 48):
            add(isa.MOVZ(6, imm, sh), "movz")
            add(isa.MOVK(6, imm, sh), "movk")

    # memory
    for off in (0, 8, 64, 4088):
        add(isa.LDR_x(1, 2, off), "ldr x")
        add(isa.STR_x(1, 2, off), "str x")
        add(isa.LDR_x(1, 31, off), "ldr x sp")
        add(isa.PRFM(3, off), "prfm")
    for off in (-16, -64, 0, 16, 504):
        add(isa.STP_pre(29, 30, 31, off), "stp pre")
        add(isa.LDP_post(29, 30, 31, off), "ldp post")
    for off in (0, 16, 32, 240, 65520):
        add(isa.LDR_q(5, 0, off), "ldr q")
        add(isa.STR_q(5, 0, off), "str q")
    for off in (0, 4, 40, 16380):
        add(isa.LDR_s(5, 0, off), "ldr s")
        add(isa.STR_s(5, 0, off), "str s")
    for rm in (1, 9, 20):
        add(isa.LDR_q_reg(5, 0, rm), "ldr q reg")

    # neon float
    trios = ((0, 1, 2), (7, 8, 9), (15, 16, 17), (29, 30, 31), (31, 0, 15))
    binops = [
        (isa.FMLA, "fmla"), (isa.FMLS, "fmls"), (isa.FMUL, "fmul"),
        (isa.FADD, "fadd"), (isa.FSUB, "fsub"), (isa.FDIV, "fdiv"),
        (isa.FMAX, "fmax"), (isa.FMIN, "fmin"), (isa.FRECPS, "frecps"),
        (isa.FRSQRTS, "frsqrts"), (isa.FCMGT, "fcmgt"), (isa.FCMGE, "fcmge"),
        (isa.VADD, "vadd"), (isa.VSUB, "vsub"), (isa.SMAX, "smax"),
        (isa.SMIN, "smin"), (isa.AND_v, "and"), (isa.ORR_v, "orr"),
        (isa.EOR_v, "eor"), (isa.BSL, "bsl"), (isa.FADDP, "faddp"),
        (isa.TRN1, "trn1"), (isa.TRN2, "trn2"),
        (isa.ZIP1_d, "zip1"), (isa.ZIP2_d, "zip2"),
    ]
    unops = [
        (isa.FNEG, "fneg"), (isa.FABS, "fabs"), (isa.FSQRT, "fsqrt"),
        (isa.FRECPE, "frecpe"), (isa.FRSQRTE, "frsqrte"), (isa.FRINTN, "frintn"),
        (isa.FCVTZS, "fcvtzs"), (isa.SCVTF, "scvtf"), (isa.MOV_v, "mov v"),
        (isa.FADDP_s, "faddp s"), (isa.FMAXV, "fmaxv"),
    ]
    for rd, rn, rm in trios:
        for fn, tag in binops:
            add(fn(rd, rn, rm), tag)
        for fn, tag in unops:
            add(fn(rd, rn), tag)
        add(isa.MOVI_zero(rd), "movi 0")
        add(isa.DUP_from_w(rd, rn), "dup w")
        for idx in range(4):
            add(isa.FMLA_lane(rd, rn, rm, idx), "fmla lane")
            add(isa.DUP_lane(rd, rn, idx), "dup lane")
    for sh in (0, 1, 7, 23, 31):
        add(isa.VSHL(3, 4, sh), "shl")

    # scalar float
    for rd, rn, rm in trios:
        add(isa.FADD_s(rd, rn, rm), "fadd s")
        add(isa.FMUL_s(rd, rn, rm), "fmul s")
        add(isa.FSUB_s(rd, rn, rm), "fsub s")
        add(isa.FDIV_s(rd, rn, rm), "fdiv s")
        add(isa.FMAX_s(rd, rn, rm), "fmax s")
        add(isa.FMADD_s(rd, rn, rm, 3), "fmadd s")
        add(isa.FSQRT_s(rd, rn), "fsqrt s")
        add(isa.FMOV_s_from_w(rd, rn), "fmov s<-w")
        add(isa.FMOV_w_from_s(rd, rn), "fmov w<-s")

    return out


def assemble(texts):
    """Assemble each text as its own instruction; return list of 32-bit words."""
    # A branch's immediate is relative to its own address, so each instruction
    # gets its own 4-byte slot and we read the words back in order.
    src = ".text\n" + "\n".join(texts) + "\n"
    with tempfile.TemporaryDirectory() as d:
        s = os.path.join(d, "a.s")
        o = os.path.join(d, "a.o")
        with open(s, "w") as f:
            f.write(src)
        r = subprocess.run(["clang", "-c", "-arch", "arm64", s, "-o", o],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("clang failed:\n" + r.stderr)
        r = subprocess.run(["otool", "-t", o], capture_output=True, text=True)
        words = []
        for line in r.stdout.splitlines():
            toks = line.split()
            if len(toks) < 2 or not re.fullmatch(r"[0-9a-f]{16}", toks[0]):
                continue
            for t in toks[1:]:
                if re.fullmatch(r"[0-9a-f]{8}", t):
                    words.append(int(t, 16))
        return words


def main():
    cs = cases()
    texts = [c[0].text for c in cs]
    try:
        ref = assemble(texts)
    except RuntimeError as e:
        print(e)
        return 1

    if len(ref) != len(cs):
        print(f"FAIL: assembled {len(ref)} words for {len(cs)} instructions "
              f"(a macro expanded to more than one instruction)")
        return 1

    bad = []
    for (ins, tag), got in zip(cs, ref):
        if ins.w != got:
            bad.append((tag, ins.text, ins.w, got))

    tags = sorted({t for _, t in cs})
    print(f"instructions checked : {len(cs)}")
    print(f"distinct forms       : {len(tags)}")
    if bad:
        print(f"MISMATCHES           : {len(bad)}")
        for tag, text, mine, theirs in bad[:40]:
            diff = mine ^ theirs
            print(f"  [{tag:10}] {text:44} kiln=0x{mine:08X} "
                  f"clang=0x{theirs:08X} xor=0x{diff:08X}")
        return 1
    print("MISMATCHES           : 0")
    print()
    print("Every instruction KILN emits is bit-identical to what Apple's")
    print("own assembler produces for the same mnemonic.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
