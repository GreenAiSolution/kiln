"""
kiln.asm - assembler layer: labels, branch fixups, and a printable listing.

The listing is not a debugging nicety. Because every Insn carries its own
assembly text, a whole generated kernel can be re-assembled by clang and
compared against our bytes - so KILN's output is verified at the kernel
level, not just instruction by instruction.
"""

import struct

from . import isa


class Asm:
    def __init__(self, name="kernel"):
        self.name = name
        self.insns = []          # list of isa.Insn (may hold placeholders)
        self.labels = {}         # name -> instruction index
        self.fixups = []         # (index, kind, cond_or_reg, label)
        self._anon = 0

    # ---------------------------------------------------------- emitting

    def emit(self, insn):
        self.insns.append(insn)
        return len(self.insns) - 1

    def __call__(self, insn):
        return self.emit(insn)

    def label(self, name=None):
        if name is None:
            self._anon += 1
            name = f"L{self._anon}"
        if name in self.labels:
            raise ValueError(f"duplicate label {name}")
        self.labels[name] = len(self.insns)
        return name

    def b(self, label):
        i = self.emit(isa.B(0))
        self.fixups.append((i, "b", None, label))

    def bcond(self, cond, label):
        i = self.emit(isa.BCOND(cond, 0))
        self.fixups.append((i, "bcond", cond, label))

    def cbnz(self, reg, label):
        i = self.emit(isa.CBNZ(reg, 0))
        self.fixups.append((i, "cbnz", reg, label))

    def cbz(self, reg, label):
        i = self.emit(isa.CBZ(reg, 0))
        self.fixups.append((i, "cbz", reg, label))

    def mov_imm(self, rd, value):
        """Materialise an arbitrary 64-bit constant."""
        if value < 0:
            raise ValueError("mov_imm takes unsigned values")
        parts = [(value >> s) & 0xFFFF for s in (0, 16, 32, 48)]
        first = True
        for idx, p in enumerate(parts):
            if p == 0 and not (first and idx == 3):
                if not first:
                    continue
                if any(parts[idx + 1:]):
                    continue
            if first:
                self.emit(isa.MOVZ(rd, p, idx * 16))
                first = False
            else:
                self.emit(isa.MOVK(rd, p, idx * 16))
        if first:
            self.emit(isa.MOVZ(rd, 0, 0))

    def mov_f32(self, vd, xtmp, value):
        """Broadcast a float32 constant into all four lanes of vd."""
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        self.mov_imm(xtmp, bits)
        self.emit(isa.DUP_from_w(vd, xtmp))

    # ---------------------------------------------------------- finishing

    def resolve(self):
        """Apply branch fixups in place. Idempotent."""
        for idx, kind, extra, label in self.fixups:
            if label not in self.labels:
                raise KeyError(f"undefined label {label}")
            delta = self.labels[label] - idx
            if kind == "b":
                self.insns[idx] = isa.B(delta)
            elif kind == "bcond":
                self.insns[idx] = isa.BCOND(extra, delta)
            elif kind == "cbnz":
                self.insns[idx] = isa.CBNZ(extra, delta)
            elif kind == "cbz":
                self.insns[idx] = isa.CBZ(extra, delta)
        self.fixups = []
        return self

    def code(self):
        self.resolve()
        return b"".join(struct.pack("<I", i.w) for i in self.insns)

    def listing(self):
        """Assembly text, with labels, that clang can re-assemble."""
        self.resolve()
        at = {}
        for name, i in self.labels.items():
            at.setdefault(i, []).append(name)
        out = []
        for i, insn in enumerate(self.insns):
            for name in at.get(i, []):
                out.append(f"{name}:")
            out.append("    " + insn.text)
        for name in at.get(len(self.insns), []):
            out.append(f"{name}:")
        return "\n".join(out)

    def __len__(self):
        return len(self.insns)

    def stats(self):
        """Instruction mix, for the scheduler's cost model and for reports."""
        mix = {}
        for insn in self.insns:
            op = insn.text.split()[0]
            mix[op] = mix.get(op, 0) + 1
        return mix
