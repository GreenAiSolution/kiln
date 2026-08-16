import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random

from kiln import ir, jit
from kiln.runtime import compile as kcompile
from kiln.lower import Schedule


def make(n, seed=1):
    r = random.Random(seed)
    return jit.Buf.of([ir.f32(r.uniform(-2, 2)) for _ in range(n)])


def check_map(name, build, n, sched):
    p = ir.Program(name)
    build(p, n)
    bufs = {}
    for nm in p.inputs:
        bufs[nm] = make(n, hash(nm) & 0xFFFF)
    for nm in p.outputs:
        if nm not in bufs:
            bufs[nm] = jit.Buf(n)
    ref = {nm: list(b.tolist()) for nm, b in bufs.items()}
    ir.run_reference(p, ref)

    c = kcompile(p, sched)
    c.run(bufs)
    got = bufs[p.outputs[0]].tolist()
    want = ref[p.outputs[0]]
    bad = sum(1 for x, y in zip(got, want) if x != y)
    err = max((abs(x - y) for x, y in zip(got, want)), default=0.0)
    print(f"{name:22} n={n:<7} U={c.kernels[0].meta['unroll']} "
          f"insns={c.insn_count():<5} exact={n - bad}/{n} maxabs={err:.3e}")
    return bad == 0


def chain(p, n):
    a, b, c = ir.load("a"), ir.load("b"), ir.load("c")
    p.map([("out", (a * b + c) * a - b)], n)


def relu_chain(p, n):
    a, b = ir.load("a"), ir.load("b")
    p.map([("out", ir.relu(a * 2.0 + b) * ir.relu(b - a))], n)


def expchain(p, n):
    a = ir.load("a")
    p.map([("out", ir.exp(a) + ir.exp(-a))], n)


def sqrtchain(p, n):
    a, b = ir.load("a"), ir.load("b")
    p.map([("out", ir.sqrt(a * a + b * b))], n)


ok = True
for sched in (Schedule(1), Schedule(2), Schedule(4), Schedule(8)):
    print(f"--- {sched}")
    for n in (4, 16, 1000, 1024, 4099):
        ok &= check_map("chain", chain, n, sched)
        ok &= check_map("relu", relu_chain, n, sched)
        ok &= check_map("exp", expchain, n, sched)
        ok &= check_map("sqrt", sqrtchain, n, sched)

print("ALL EXACT" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
