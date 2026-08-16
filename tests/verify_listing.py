"""
Whole-kernel differential test, and the dependency audit.

tests/verify_isa.py checks instructions one at a time. That will not catch a
branch fixup computing the wrong displacement, or a label resolving to the
wrong index - bugs that live between instructions rather than inside them.

So this takes complete generated kernels, prints them as assembly, hands the
whole listing to clang, and compares the object code byte for byte against
what KILN put in memory. Labels, branch distances, loop structure and all.

Then it walks every module under kiln/ and asserts that the only things they
import are the Python standard library. The speed claims are only interesting
if nothing is quietly doing the work for us.
"""

import ast
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kiln import gemm, ir, nn, transpose                      # noqa: E402
from kiln.lower import Schedule                               # noqa: E402
from kiln.runtime import compile as kcompile                  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KILN = os.path.join(HERE, "kiln")

STDLIB = set(sys.stdlib_module_names)


def assemble(listing):
    with tempfile.TemporaryDirectory() as d:
        s, o = os.path.join(d, "k.s"), os.path.join(d, "k.o")
        with open(s, "w") as f:
            f.write(".text\n" + listing + "\n")
        r = subprocess.run(["clang", "-c", "-arch", "arm64", s, "-o", o],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr)
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


def kernels():
    out = []

    def prog(name, build, n, sched):
        p = ir.Program(name)
        build(p, n)
        ir.contract_program(p)
        c = kcompile(p, sched, contract=False)
        for i, k in enumerate(c.kernels):
            out.append((f"{name}[{i}] n={n} {sched.key()}", k.asm))

    def chain(p, n):
        a, b, c = ir.load("a"), ir.load("b"), ir.load("c")
        p.map([("out", (a * b + c) * a - b)], n)

    def softmax(p, n):
        x = ir.load("a")
        p.reduce("m", "max", x, n)
        e = ir.exp(x - ir.sarg("m"))
        p.reduce("s", "sum", e, n)
        p.map([("out", e * ir.recip(ir.sarg("s")))], n)

    def layernorm(p, n):
        x = ir.load("a")
        p.reduce("s1", "sum", x, n)
        mu = ir.sarg("s1") * ir.f32(1.0 / n)
        p.reduce("s2", "sum", (x - mu) * (x - mu), n)
        inv = ir.rsqrt(ir.sarg("s2") * ir.f32(1.0 / n) + 1e-5)
        p.map([("out", (x - mu) * inv)], n)

    def gelu(p, n):
        p.map([("out", ir.gelu(ir.load("a")))], n)

    def steps(p, n):
        p.map([("dz", ir.load("dy") * ir.step(ir.load("y")))], n)

    for n in (1000, 1024, 4099, 1 << 20):
        for sched in (Schedule(1), Schedule(4), Schedule(8),
                      Schedule(6, prefetch=256)):
            prog("chain", chain, n, sched)
            prog("softmax", softmax, n, sched)
            prog("layernorm", layernorm, n, sched)
            prog("gelu", gelu, n, sched)
            prog("step", steps, n, sched)

    for M, N, K in ((64, 64, 64), (128, 256, 128), (256, 128, 64)):
        out.append((f"gemm {M}x{N}x{K}", gemm.gemm(M, N, K).asm))
        out.append((f"gemm+acc {M}x{N}x{K}",
                    gemm.gemm(M, N, K, accumulate=True).asm))
    for R, C in ((64, 32), (128, 256), (512, 512)):
        out.append((f"transpose {R}x{C}", transpose.transpose(R, C).asm))
    for B, D in ((128, 128), (64, 32), (8, 512)):
        out.append((f"bias_relu {B}x{D}", nn.build_bias_act(B, D, True)))
        out.append((f"bias {B}x{D}", nn.build_bias_act(B, D, False)))
    return out


def check_listings():
    ks = kernels()
    bad, total_insns = [], 0
    for name, asm in ks:
        mine = [i.w for i in asm.resolve().insns]
        total_insns += len(mine)
        try:
            theirs = assemble(asm.listing())
        except RuntimeError as e:
            bad.append((name, f"clang rejected the listing: {e}"))
            continue
        if len(mine) != len(theirs):
            bad.append((name, f"length {len(mine)} vs {len(theirs)}"))
            continue
        for i, (a, b) in enumerate(zip(mine, theirs)):
            if a != b:
                bad.append((name, f"insn {i}: 0x{a:08X} vs 0x{b:08X} "
                                  f"({asm.insns[i].text})"))
                break
    print(f"whole kernels re-assembled : {len(ks)}")
    print(f"instructions compared      : {total_insns:,}")
    print(f"mismatches                 : {len(bad)}")
    for n, m in bad[:10]:
        print(f"   {n}: {m}")
    return not bad


def check_imports():
    offenders, mods = [], []
    for fn in sorted(os.listdir(KILN)):
        if not fn.endswith(".py"):
            continue
        mods.append(fn)
        path = os.path.join(KILN, fn)
        with open(path) as f:
            tree = ast.parse(f.read(), path)
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:            # relative import inside kiln
                    continue
                names = [(node.module or "").split(".")[0]]
            for nm in names:
                if nm and nm not in STDLIB:
                    offenders.append((fn, nm))
    print(f"\nmodules under kiln/        : {len(mods)}")
    print(f"non-stdlib imports         : {len(offenders)}")
    for f, n in offenders:
        print(f"   {f} imports {n}")
    if not offenders:
        print("kiln/ imports nothing outside the Python standard library.")
    return not offenders


def main():
    print("Whole-kernel verification")
    print("=" * 74)
    a = check_listings()
    b = check_imports()
    print()
    if a and b:
        print("PASS")
        return 0
    print("FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
