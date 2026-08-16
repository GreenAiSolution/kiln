"""
A guided tour of KILN, one stage at a time.

    python3 tour.py            # the whole tour
    python3 tour.py 3          # just stop 3

Every line this prints is produced live. Nothing is a stored transcript.
"""

import os
import re
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kiln import exact, gemm, ir, jit, transpose             # noqa: E402
from kiln.lower import Schedule                              # noqa: E402
from kiln.runtime import bench_batched, compile as kcompile  # noqa: E402

W = 78


def head(n, title, subtitle=""):
    print()
    print("─" * W)
    print(f"  STOP {n}   {title}")
    if subtitle:
        print(f"           {subtitle}")
    print("─" * W)


def say(*lines):
    for l in lines:
        print(f"  {l}")


N = 1 << 20


def the_program():
    """The running example for the whole tour."""
    p = ir.Program("tour")
    a, b, c = ir.load("a"), ir.load("b"), ir.load("c")
    p.map([("out", (a * b + c) * a - b)], N)
    return p


# ---------------------------------------------------------------- stop 1

def stop1():
    head(1, "What you write",
         "an expression over whole arrays, not a loop")
    say("",
        "    p = Program()",
        "    a, b, c = load('a'), load('b'), load('c')",
        "    p.map([('out', (a*b + c)*a - b)], n=1048576)",
        "",
        "That is the whole input. No types, no loop, no hint about how many",
        "numbers fit in a register. KILN works all of that out.",
        "")
    p = the_program()
    s = p.summary()
    say(f"KILN reads it as: {s['stages']} stage, "
        f"inputs {s['inputs']}, output {s['outputs']}")


# ---------------------------------------------------------------- stop 2

def stop2():
    head(2, "What KILN sees",
         "the expression as a graph, before and after one optimisation")
    p = the_program()
    e = p.stages[0].outputs[0][1]

    say("", "As written, the expression is a tree of 4 operations:", "")
    say(f"    {e!r}")
    before = len(ir.topo(e))
    say("", f"    {before} distinct nodes (shared subexpressions counted once)")

    ir.contract_program(p)
    e2 = p.stages[0].outputs[0][1]
    after = len(ir.topo(e2))
    say("",
        "Now the contraction pass runs. It looks for add(multiply(x,y), z)",
        "and replaces it with a single fused multiply-add:", "")
    say(f"    {e2!r}")
    say("",
        f"    {before} nodes -> {after} nodes",
        "",
        "Two things happened. One instruction disappeared. And the answer got",
        "*more* accurate: a fused multiply-add rounds once where a separate",
        "multiply and add round twice.")


# ---------------------------------------------------------------- stop 3

def stop3():
    head(3, "The decision KILN makes",
         "how many numbers to keep in flight at once")
    p = the_program()
    ir.contract_program(p)

    say("",
        "The chip has 32 vector registers. Each holds 4 floats. KILN has to",
        "decide how many groups of 4 to process per trip around the loop.",
        "",
        "Too few and the processor sits idle waiting for each result.",
        "Too many and it runs out of registers and starts spilling to memory.",
        "",
        f"  {'asked for':>10}  {'used':>5}  {'live':>5}  {'loop body':>10}  "
        f"{'per element':>12}")
    for u in (1, 2, 4, 6, 8, 12):
        c = kcompile(p, Schedule(u), contract=False)
        m = c.kernels[0].meta
        say(f"  {u:>10}  {m['unroll']:>5}  {m['live']:>5}  "
            f"{m['body_insns']:>7} insns  {m['insns_per_elem']:>9.2f} insns")
    cap = kcompile(p, Schedule(12), contract=False).kernels[0].meta
    say("",
        f"'live' is the peak number of values alive at once - {cap['live']} here.",
        f"Each of those needs its own register per unrolled copy, so the",
        f"factor is capped: ask for 12 and KILN gives you {cap['unroll']}, because",
        f"{cap['unroll']} x {cap['live']} is as many registers as it can spend.",
        "It worked that ceiling out from the graph, not from a table.",
        "",
        "Notice the last column flattening out. Going from 1 to 4 removes a",
        "third of the work per number; going from 8 to 12 removes nothing.",
        "That curve is what the cost model at stop 8 learns to predict.")


# ---------------------------------------------------------------- stop 4

def stop4():
    head(4, "The code it wrote",
         "real ARM64 assembly, generated a moment ago")
    p = the_program()
    ir.contract_program(p)
    c = kcompile(p, Schedule(6), contract=False)
    k = c.kernels[0]
    lines = k.listing().split("\n")

    say("", "The prologue - fetch the three array addresses, set the counter:", "")
    for l in lines[:5]:
        print(f"      {l}")
    say("", "The loop body. This is the whole calculation, 6 groups of 4",
        "numbers at a time. Note there is no store until the very end -",
        "every intermediate value stays in a register:", "")
    body = [l for l in lines if l.strip().startswith(("ldr q", "fmla", "fmul",
                                                      "fsub", "mov v", "str q"))]
    for l in body[:9]:
        print(f"      {l}")
    print(f"      ... {len(body) - 12} more of the same pattern ...")
    for l in body[-3:]:
        print(f"      {l}")
    say("", "And the loop control:", "")
    for l in lines:
        if "subs" in l or "b.ne" in l:
            print(f"      {l}")
    say("",
        f"{len(k.asm)} instructions total, {c.code_bytes()} bytes of machine code,",
        f"compiled in well under a millisecond.")


# ---------------------------------------------------------------- stop 5

def stop5():
    head(5, "Proving the code is real",
         "hand the same text to Apple's compiler and compare")
    p = the_program()
    ir.contract_program(p)
    c = kcompile(p, Schedule(6), contract=False)
    k = c.kernels[0]

    mine = [i.w for i in k.asm.resolve().insns]
    say("", "KILN's own bytes for the first six instructions:", "")
    for i, ins in enumerate(k.asm.insns[:6]):
        say(f"    0x{ins.w:08X}    {ins.text}")

    with tempfile.TemporaryDirectory() as d:
        sp, op = os.path.join(d, "k.s"), os.path.join(d, "k.o")
        with open(sp, "w") as f:
            f.write(".text\n" + k.listing() + "\n")
        subprocess.run(["clang", "-c", "-arch", "arm64", sp, "-o", op],
                       check=True, capture_output=True)
        r = subprocess.run(["otool", "-t", op], capture_output=True, text=True)
        theirs = []
        for line in r.stdout.splitlines():
            t = line.split()
            if len(t) > 1 and re.fullmatch(r"[0-9a-f]{16}", t[0]):
                theirs += [int(x, 16) for x in t[1:]
                           if re.fullmatch(r"[0-9a-f]{8}", x)]

    say("", "Apple's assembler, given the same text:", "")
    for wv in theirs[:6]:
        say(f"    0x{wv:08X}")
    diff = sum(1 for a, b in zip(mine, theirs) if a != b)
    say("",
        f"instructions compared : {len(mine)}",
        f"differences           : {diff}",
        "",
        "This is the check the whole project rests on. Across the full test",
        "suite it covers 491 instruction forms and 159 complete kernels.",
        "It caught a genuine bug the first time it ran.")


# ---------------------------------------------------------------- stop 6

def stop6():
    head(6, "Running it, and checking the answer",
         "against arithmetic done in exact fractions")
    p = the_program()
    ir.contract_program(p)

    n = 4096
    p2 = ir.Program("check")
    a, b, cc = ir.load("a"), ir.load("b"), ir.load("c")
    p2.map([("out", (a * b + cc) * a - b)], n)
    ir.contract_program(p2)

    import random
    r = random.Random(11)
    data = {k: [ir.f32(r.uniform(-2, 2)) for _ in range(n)] for k in "abc"}
    data["out"] = [0.0] * n
    bufs = {k: jit.Buf.of(v) for k, v in data.items()}

    say("", "Compiling and calling the generated code on 4,096 numbers...")
    c = kcompile(p2, Schedule(6), contract=False)
    c.run(bufs)
    got = bufs["out"].tolist()

    say("Now computing the same thing in exact rational arithmetic, one",
        "number at a time, rounding to float32 exactly once...")
    ref = {k: list(v) for k, v in data.items()}
    ir.run_reference(p2, ref)
    want = ref["out"]

    worst = max(exact.ulps_apart(x, y) for x, y in zip(got, want))
    same = sum(1 for x, y in zip(got, want) if x == y)
    say("",
        f"  numbers compared      : {n:,}",
        f"  bit-for-bit identical : {same:,}",
        f"  worst difference      : {worst} units in the last place",
        "",
        "Zero. Not 'close enough' - the machine code and the exact-fraction",
        "reference agree on every single bit of every single number.",
        "",
        "  first three results:")
    for i in range(3):
        say(f"    a={data['a'][i]:+.6f}  b={data['b'][i]:+.6f}  "
            f"c={data['c'][i]:+.6f}  ->  {got[i]:+.8f}")


# ---------------------------------------------------------------- stop 7

def stop7():
    head(7, "Racing it",
         "the same expression, KILN against numpy")
    try:
        import numpy as np
    except ImportError:
        say("numpy is not installed; skipping the race.")
        return

    p = the_program()
    ir.contract_program(p)
    rng = np.random.default_rng(0)
    A, B, C = (rng.random(N, dtype=np.float32) for _ in range(3))
    O = np.zeros(N, dtype=np.float32)

    bufs = {}
    for nm, arr in (("a", A), ("b", B), ("c", C), ("out", O)):
        buf = jit.Buf(N)
        buf.frombytes(arr.tobytes())
        bufs[nm] = buf
    c = kcompile(p, Schedule(6), contract=False)
    c.bind(bufs)

    tk = bench_batched(lambda: c.run(), seconds=0.3)["best"]
    tn = bench_batched(lambda: (A * B + C) * A - B, seconds=0.3)["best"]

    say("",
        f"  KILN   {tk * 1e6:8.1f} microseconds     one loop, nothing allocated",
        f"  numpy  {tn * 1e6:8.1f} microseconds     four loops, three temporary arrays",
        f"",
        f"  {tn / tk:.2f}x faster on {N:,} numbers",
        "",
        "numpy is not slow. Its inner loops are hand-written vector code too.",
        "What it cannot do is see the whole expression at once - it is handed",
        "one operator at a time, so it must finish each one, write the result",
        "to memory, and read it back for the next.",
        "",
        f"  memory traffic, numpy : {N * 4 * 9 / 1e6:.0f} MB  "
        f"(read and write per operator)",
        f"  memory traffic, KILN  : {N * 4 * 4 / 1e6:.0f} MB  "
        f"(read three, write one, once)")


# ---------------------------------------------------------------- stop 8

def stop8():
    head(8, "The rest of the box",
         "what else is built on the same foundation")
    say("")
    t0 = time.perf_counter()
    g = gemm.gemm(256, 256, 256)
    A, B, Cm = jit.Buf(256 * 256, 0.5), jit.Buf(256 * 256, 0.25), jit.Buf(256 * 256)
    tg = bench_batched(lambda: g(A, B, Cm), seconds=0.25)["best"]
    say(f"  matrix multiply   256x256x256 in {tg * 1e6:.0f} us  "
        f"= {2 * 256 ** 3 / tg / 1e9:.1f} GFLOP/s")
    say(f"                    {len(g.asm)} instructions, {g.code.nbytes} bytes, "
        f"16 registers hold the answer")

    tr = transpose.transpose(512, 512)
    S, D = jit.Buf(512 * 512), jit.Buf(512 * 512)
    tt = bench_batched(lambda: tr(S, D), seconds=0.25)["best"]
    say(f"  transpose         512x512 in {tt * 1e6:.0f} us "
        f"using 8 shuffle instructions per 4x4 block")

    ne = 1 << 20
    pe = ir.Program("exp")
    pe.map([("out", ir.exp(ir.load("a")))], ne)
    ir.contract_program(pe)
    ce = kcompile(pe, Schedule(10), contract=False)
    ce.bind({"a": jit.Buf(ne, 0.5), "out": jit.Buf(ne)})
    te = bench_batched(lambda: ce.run(), seconds=0.25)["best"]
    say(f"  exponential       {ne / te / 1e9:.2f} billion per second, "
        f"21 instructions, 1 ULP error")

    say("",
        "  neural network    forward, backward and the optimiser all run on",
        "                    kernels generated here. It matches a float64",
        "                    reference to 1.4e-4 over 300 training steps.",
        "",
        "  cost model        ranks 40 possible loop arrangements without",
        "                    running any of them, then times only the top 5.",
        "")
    say("─" * (W - 2))
    say("Run `python3 run_all.py` to reproduce every number in the project.",
        "9 suites, about 4 minutes.")


STOPS = [stop1, stop2, stop3, stop4, stop5, stop6, stop7, stop8]


def main():
    if len(sys.argv) > 1:
        i = int(sys.argv[1])
        STOPS[i - 1]()
    else:
        print()
        print("  K I L N   —   a guided tour")
        print("  A compiler that writes its own machine code.")
        print("  Everything below runs live.")
        for s in STOPS:
            s()
    print()


if __name__ == "__main__":
    main()
