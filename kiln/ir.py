"""
kiln.ir - the intermediate representation.

An expression is a DAG over float32 elements. A Program is a short list of
stages, each of which is either a map (walk N elements, write outputs) or a
reduce (walk N elements, produce one scalar). Everything inside a stage is
fused: intermediates live in vector registers and never touch memory. That
single property is where most of the speed comes from - a library has to
materialise every temporary, a compiler does not.

Expressions are hash-consed on construction, so common subexpressions are
shared by identity before any pass runs.

Pure standard library.
"""

import math
import struct

from . import exact

# ---------------------------------------------------------------- helpers


def f32(x):
    """Round to nearest float32, exactly as the hardware would."""
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


BINARY = {"add", "sub", "mul", "div", "max", "min"}
UNARY = {"neg", "abs", "sqrt", "recip", "rsqrt", "exp", "step"}
LEAF = {"load", "const", "sarg"}
TERNARY = {"fma"}          # fma(a, b, c) = a*b + c, one rounding


class Expr:
    __slots__ = ("op", "args", "attr", "_h", "id")
    _pool = {}
    _next_id = [0]

    def __new__(cls, op, args=(), attr=None):
        key = (op, tuple(a.id for a in args), attr)
        hit = cls._pool.get(key)
        if hit is not None:
            return hit
        self = object.__new__(cls)
        self.op = op
        self.args = tuple(args)
        self.attr = attr
        self._h = hash(key)
        self.id = cls._next_id[0]
        cls._next_id[0] += 1
        cls._pool[key] = self
        return self

    def __hash__(self):
        return self._h

    def __repr__(self):
        if self.op == "const":
            return f"{self.attr:g}"
        if self.op == "load":
            return f"{self.attr}[i]"
        if self.op == "sarg":
            return f"${self.attr}"
        return f"{self.op}({', '.join(map(repr, self.args))})"

    # ------------------------------------------------------ operator sugar

    def _lift(self, o):
        if isinstance(o, Expr):
            return o
        return const(o)

    def __add__(self, o):
        return binop("add", self, self._lift(o))

    __radd__ = __add__

    def __sub__(self, o):
        return binop("sub", self, self._lift(o))

    def __rsub__(self, o):
        return binop("sub", self._lift(o), self)

    def __mul__(self, o):
        return binop("mul", self, self._lift(o))

    __rmul__ = __mul__

    def __truediv__(self, o):
        return binop("div", self, self._lift(o))

    def __rtruediv__(self, o):
        return binop("div", self._lift(o), self)

    def __neg__(self):
        return Expr("neg", (self,))

    def __pow__(self, k):
        if k == 2:
            return self * self
        if k == 3:
            return self * self * self
        if k == 0.5:
            return Expr("sqrt", (self,))
        raise ValueError("only powers 2, 3 and 0.5 are supported")


def const(v):
    return Expr("const", (), f32(v))


def load(name):
    return Expr("load", (), name)


def sarg(name):
    """A scalar produced by an earlier stage and passed in at run time."""
    return Expr("sarg", (), name)


def binop(op, a, b):
    return Expr(op, (a, b))


def emax(a, b):
    return binop("max", a, b if isinstance(b, Expr) else const(b))


def emin(a, b):
    return binop("min", a, b if isinstance(b, Expr) else const(b))


def exp(a):
    return Expr("exp", (a,))


def sqrt(a):
    return Expr("sqrt", (a,))


def recip(a):
    return Expr("recip", (a,))


def rsqrt(a):
    return Expr("rsqrt", (a,))


def relu(a):
    return emax(a, 0.0)


def step(a):
    """1.0 where a > 0, else 0.0 - the derivative of relu."""
    return Expr("step", (a,))


def sigmoid(a):
    return recip(const(1.0) + exp(-a))


def tanh(a):
    e = exp(a * 2.0)
    return (e - 1.0) * recip(e + 1.0)


def gelu(a):
    """tanh approximation, the one transformers actually use."""
    inner = 0.7978845608028654 * (a + 0.044715 * (a * a * a))
    return a * 0.5 * (tanh(inner) + 1.0)


# ------------------------------------------------------------------ stages


class Map:
    """out_k[i] = expr_k(i) for i in range(n)."""

    kind = "map"

    def __init__(self, outputs, n):
        self.outputs = list(outputs)      # [(buffer_name, Expr)]
        self.n = n

    def exprs(self):
        return [e for _, e in self.outputs]


class Reduce:
    """acc = fold(expr(i)) for i in range(n); result bound to a scalar name."""

    kind = "reduce"

    def __init__(self, name, how, expr, n):
        assert how in ("sum", "max")
        self.name = name
        self.how = how
        self.expr = expr
        self.n = n

    def exprs(self):
        return [self.expr]


class Program:
    """A named list of stages plus the buffers they touch."""

    def __init__(self, name):
        self.name = name
        self.stages = []
        self.inputs = []        # buffer names read
        self.outputs = []       # buffer names written
        self.scalars = []       # scalar names produced by reduce stages

    def map(self, outputs, n):
        st = Map(outputs, n)
        self.stages.append(st)
        self._collect(st)
        for name, _ in st.outputs:
            if name not in self.outputs:
                self.outputs.append(name)
        return st

    def reduce(self, name, how, expr, n):
        st = Reduce(name, how, expr, n)
        self.stages.append(st)
        self._collect(st)
        self.scalars.append(name)
        return st

    def _collect(self, st):
        for e in st.exprs():
            for node in walk(e):
                if node.op == "load" and node.attr not in self.inputs:
                    self.inputs.append(node.attr)

    def buffers(self):
        names = list(self.inputs)
        for o in self.outputs:
            if o not in names:
                names.append(o)
        return names

    def summary(self):
        return {
            "stages": len(self.stages),
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "scalars": list(self.scalars),
            "nodes": sum(len(topo(e)) for st in self.stages for e in st.exprs()),
        }


def walk(e, seen=None):
    """Every distinct node under e."""
    if seen is None:
        seen = set()
    if e.id in seen:
        return
    seen.add(e.id)
    yield e
    for a in e.args:
        yield from walk(a, seen)


def topo(*roots):
    """Nodes in evaluation order, each appearing once."""
    order, seen = [], set()

    def visit(e):
        if e.id in seen:
            return
        seen.add(e.id)
        for a in e.args:
            visit(a)
        order.append(e)

    for r in roots:
        visit(r)
    return order


# ------------------------------------------------- optimisation: contraction

def contract_fma(e, memo=None):
    """add(mul(a,b), c) -> fma(a,b,c).

    Worth doing explicitly: it halves the instruction count of most kernels
    and removes one rounding step per operation, so the fused result is
    closer to the exact answer than the two-step version.
    """
    if memo is None:
        memo = {}
    if e.id in memo:
        return memo[e.id]
    args = tuple(contract_fma(a, memo) for a in e.args)
    if e.op == "add":
        a, b = args
        if a.op == "mul":
            out = Expr("fma", (a.args[0], a.args[1], b))
            memo[e.id] = out
            return out
        if b.op == "mul":
            out = Expr("fma", (b.args[0], b.args[1], a))
            memo[e.id] = out
            return out
    out = Expr(e.op, args, e.attr) if args != e.args else e
    memo[e.id] = out
    return out


def contract_program(p):
    for st in p.stages:
        if st.kind == "map":
            st.outputs = [(n, contract_fma(e)) for n, e in st.outputs]
        else:
            st.expr = contract_fma(st.expr)
    return p


# ------------------------------------------------------ reference evaluator

def eval_expr(e, i, bufs, scalars, cache=None):
    """Ground truth, in float32, one element at a time. Slow on purpose:
    everything the compiler emits is checked against this."""
    if cache is None:
        cache = {}
    key = e.id
    if key in cache:
        return cache[key]
    op = e.op
    if op == "const":
        v = e.attr
    elif op == "load":
        v = bufs[e.attr][i]
    elif op == "sarg":
        v = scalars[e.attr]
    else:
        a = [eval_expr(x, i, bufs, scalars, cache) for x in e.args]
        if op == "add":
            v = exact.add32(a[0], a[1])
        elif op == "sub":
            v = exact.sub32(a[0], a[1])
        elif op == "mul":
            v = exact.mul32(a[0], a[1])
        elif op == "div":
            v = exact.div32(a[0], a[1])
        elif op == "max":
            v = max(a[0], a[1])
        elif op == "min":
            v = min(a[0], a[1])
        elif op == "neg":
            v = -a[0]
        elif op == "abs":
            v = abs(a[0])
        elif op == "sqrt":
            v = exact.sqrt32(a[0])
        elif op == "recip":
            # KILN lowers this to Newton-Raphson, which is not correctly
            # rounded; the reference gives the true answer and the tests
            # report the gap rather than hiding it.
            v = exact.div32(1.0, a[0])
        elif op == "rsqrt":
            v = exact.div32(1.0, exact.sqrt32(a[0]))
        elif op == "step":
            v = 1.0 if a[0] > 0.0 else 0.0
        elif op == "exp":
            v = f32(math.exp(a[0]))
        elif op == "fma":
            v = exact.fma32(a[0], a[1], a[2])
        else:
            raise NotImplementedError(op)
    cache[key] = v
    return v


def run_reference(prog, bufs, use_scalars=None):
    """Execute a Program in pure Python.

    Sums are accumulated with math.fsum, so the reduce result is the exact
    mathematical sum rounded once - the best any float32 answer could be.
    That makes it a yardstick rather than a rival ordering; comparing KILN
    against some *other* arbitrary summation order would measure nothing.

    Pass use_scalars to force the scalars produced by an earlier run (e.g.
    KILN's own), which isolates each stage: a map stage can then be checked
    on exactly the inputs the machine code saw.

    Returns (scalars, conditioning) where conditioning[name] is the sum of
    absolute terms - the quantity every summation error bound is relative to.
    """
    scalars = {}         # what downstream stages see
    ideal = {}           # the best-possible value, for scoring
    conditioning = {}
    for st in prog.stages:
        if st.kind == "map":
            outs = {n: [0.0] * st.n for n, _ in st.outputs}
            for i in range(st.n):
                cache = {}
                for n, e in st.outputs:
                    outs[n][i] = eval_expr(e, i, bufs, scalars, cache)
            for n, vals in outs.items():
                bufs[n][:] = vals
        elif st.how == "sum":
            terms = [eval_expr(st.expr, i, bufs, scalars, {})
                     for i in range(st.n)]
            conditioning[st.name] = math.fsum(abs(t) for t in terms)
            scalars[st.name] = f32(math.fsum(terms))
        else:
            acc = -math.inf
            for i in range(st.n):
                acc = max(acc, eval_expr(st.expr, i, bufs, scalars, {}))
            conditioning[st.name] = abs(acc)
            scalars[st.name] = acc
        if st.kind == "reduce":
            ideal[st.name] = scalars[st.name]
            if use_scalars is not None:
                scalars[st.name] = use_scalars[st.name]
    return ideal, conditioning
