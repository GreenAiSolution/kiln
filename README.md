# KILN

**A tensor compiler that writes its own machine code.**

You describe a computation. KILN analyses it, decides how to schedule it,
emits ARM64 NEON instructions one 32-bit word at a time, writes them into
executable memory, and calls them. No LLVM, no assembler, no C compiler, no
numpy, no libraries of any kind — 3,275 lines of the Python standard library.

It is fast because it fuses. It is trustworthy because every instruction it
emits is checked against Apple's own assembler, and every number it computes
is checked against exact rational arithmetic.

**[Read the illustrated writeup →](https://greenaisolution.github.io/kiln/)**
· or run `python3 tour.py` and watch the compiler work, one stage at a time.

```python
from kiln import ir, jit
from kiln.runtime import compile

p = ir.Program("chain")
a, b, c = ir.load("a"), ir.load("b"), ir.load("c")
p.map([("out", (a * b + c) * a - b)], n=1 << 20)

fn = compile(p)                 # 0.6 ms: analyse, schedule, emit, load
fn.run({"a": A, "b": B, "c": C, "out": OUT})
```

That expression compiles to **one loop** that touches each element once and
allocates nothing. numpy has to walk memory four times and allocate three
temporaries, because a library sees four separate operators and a compiler
sees one expression. On 16 M elements that is the difference between 12.5 ms
and 4.3 ms.

---

## The headline numbers

All measured on an Apple M1 Max, single core, float32, and all taken from one
run whose full transcript ships in [`results/run_all.txt`](results/run_all.txt).
Reproduce with `python3 run_all.py`.

The speedup rows move a little between runs — the median lands at 2.7–2.9×.
The variance is on numpy's side: its temporaries are 64 MB each at the largest
size, so its timings depend on how the page faults fall.

One measurement bug is worth naming, because it was mine. The first version of
`bench/roofline.py` reported this core's peak at 85.6 GFLOP/s — 3.32 fused
multiply-adds per cycle. The real number is **103.3 GFLOP/s, exactly 4.00 per
cycle**. Two causes: the peak kernel was the first thing timed in a fresh
process, on a core that had not yet ramped its clock; and a ceiling is a
maximum over attempts, but only one kernel shape was tried. Every benchmark
now spins the core up first, and the peak is the best of five shapes. The
ratios barely moved — numerator and denominator were both understated — but
the absolute numbers were wrong and are now right.

| | |
|---|---|
| Instructions verified bit-identical to Apple's assembler | **491 / 491** |
| Whole kernels re-assembled and compared | **159**, 15,732 instructions, 0 mismatches |
| Kernels numerically verified | **1,386**, 9,075,402 elements |
| Kernels required to be bit-exact that were bit-exact | **7 of 7 families, 0 ULP** |
| Speedup vs idiomatic numpy | **2.9× median, 8.2× best** |
| Speedup vs numpy with preallocated `out=` | **2.2× median, 4.7× best** |
| Matrix multiply, % of this core's measured NEON ceiling | **97.6%** |
| Vectorised `exp` accuracy | **1 ULP max**, 91.3% bit-identical to libm |
| Vectorised `tanh` accuracy | **2 ULP max** (254 before it was fixed) |
| Vectorised `exp` throughput | **3.1 billion/second**, one core |
| Neural network trained on emitted code, vs float64 reference | **1.4 × 10⁻⁴** relative, 300 steps |
| Non-standard-library imports under `kiln/` | **0** |

---

## What is actually in here

### 1. An ARM64 assembler that does not trust itself

`kiln/isa.py` encodes 79 instruction forms. Every encoder returns both the
32-bit machine word *and* the assembly text that word is supposed to mean.
`tests/verify_isa.py` hands all 491 of those texts to `clang`, reads the
object code back with `otool`, and compares byte for byte.

This is the load-bearing test. It found a real bug on the first run: `LDR Q,
[Xn, Xm, LSL #4]` had one wrong bit in its option field. Everything else in
this project rests on that check passing.

`tests/verify_listing.py` goes further and re-assembles 159 *complete
kernels*, which catches the class of bug the instruction-level test cannot:
branch displacements and label resolution.

### 2. A fusing compiler

`kiln/lower.py`. The passes, in order:

1. **contract** — `add(mul(a,b), c)` becomes a single fused multiply-add.
   Halves the instruction count of most kernels, and removes a rounding step,
   so the fused result is *closer* to the exact answer than the two-step one.
2. **liveness** — when each intermediate value is born and when it dies.
3. **unroll** — how many 4-wide vectors to keep in flight, chosen from the
   register budget: 32 NEON registers, minus hoisted constants, minus
   accumulators, divided by the peak live count.
4. **allocate** — linear scan over the register file.
5. **specialise** — loop trip counts are compile-time constants. A JIT knows
   the shape it was called with, so there is no reason to compute it at
   runtime.
6. **emit** — node-major, so the unrolled copies interleave and the
   out-of-order core always has independent work.

The ragged tail of an array that is not a multiple of four is handled by
recomputing the last whole vector at an overlap — valid precisely because
every operation in the expression is a pure function of its inputs, and
skipped when an output buffer aliases an input.

### 3. A vectorised `exp` in 21 instructions

There is no vector exponential on ARM, and calling `libm` would break the
loop. So `kiln/vecexp.py` emits its own: clamp, range-reduce against a split
`ln 2` so the subtraction stays exact, evaluate a degree-6 polynomial by
Horner in fused multiply-adds, then multiply by 2ⁿ by *adding n to the
exponent field* — an integer shift and an integer add instead of a multiply.

The polynomial is not a Taylor series. `tools/fit_exp.py` implements Remez
exchange to find the minimax polynomial — the one whose worst error over the
interval is as small as it can be. A Taylor series is optimal at a single
point and wastes accuracy everywhere else; at degree 6 the minimax fit is
**50× more accurate** than the Taylor one.

Measured on hardware over 900,000 sample points: **1 ULP maximum error,
91.3% of results bit-identical to the C library.**

### 4. A matrix multiply at 97.6% of the machine's ceiling

`kiln/gemm.py` holds an 8×8 block of the output in 16 vector registers and
never spills it. Each inner step issues 16 fused multiply-adds against 2
loads, using `FMLA`'s lane-indexed form to broadcast values straight out of a
register — which is why there is no packing pass: both operands are read in
the layout they already have.

`bench/roofline.py` first measures what this core can actually do, using a
kernel of nothing but independent FMAs (**103.3 GFLOP/s**, exactly 4.00 per
cycle) and a kernel of nothing but streaming loads (**61.2 GB/s**). Against
that measured ceiling:

| shape | plain | blocked | % of ceiling |
|---|---|---|---|
| 64³ | 95.7 | 94.5 | 92.7% |
| 128³ | 100.8 | 100.7 | **97.6%** |
| 256³ | 98.4 | 98.3 | 95.3% |
| 512³ | 97.5 | 96.2 | 94.4% |
| 1024³ | 87.8 | 87.1 | 85.1% |
| 2048³ | 16.2 | **77.2** | 74.8% |

Getting the ceiling right took two tries. Measuring FMA throughput with only
8 independent accumulator chains reports 2.00 per cycle, and 12 reports 3.00 —
those are measurements of FMA *latency*, not throughput. Only at 16 chains
does the real 4.00 appear.

Blocking is not a free win everywhere: below 2048³ it is a wash or a shade
behind, because the problem already fits in cache. It matters at the top end.
2048³ collapses to 16.2 GFLOP/s without it — 16% of the ceiling — because a
column strip of B is re-streamed from DRAM for every row panel of A.
Holding a cache-sized block of B resident instead gives **77.2**. That 4.8× is
the single largest measured improvement in the project, and the inner loop's
instructions are byte-for-byte identical before and after. Only the order
changed. The unblocked figure is the least stable number here — it is cache
thrashing, so it swings between about 16 and 19 GFLOP/s run to run.

**What this does not beat.** Apple's Accelerate reaches ~2,100 GFLOP/s here,
roughly 20× more. Not by writing better NEON — by dispatching to AMX, an
on-die matrix coprocessor whose instruction encoding Apple does not publish.
No sequence of documented ARM instructions reaches it. That is the real
ceiling on "from scratch" on this chip, and it is worth naming rather than
hiding.

### 5. A cost model that learns this machine

`kiln/tune.py`. Forty candidate schedules per kernel, and the best one moves
with the size of the data and the shape of the expression — the spread
between the best and worst schedule has a median of 2.0× and reaches
**16.2×** on the worst kernel, so choosing badly is expensive.

Measuring all forty is correct and slow. So KILN fits ridge regression on
features of the *generated code* — instructions per element, loads per
element, register pressure, leftover fraction, loop trip count, and their
interactions with the schedule knobs — trained on measurements from this
machine, predicting the relative cost of a schedule.

`bench/autotune_study.py` scores it by **leave-one-kernel-out** cross
validation: fitted on every other kernel, then asked to schedule one it has
never seen. Compiling a candidate costs half a millisecond; timing one costs
far more. So the model ranks all 40 and only 5 get timed.

| | |
|---|---|
| guided / exhaustive-best | **1.019× median**, 1.38× worst |
| within 2% of the true optimum | 27 / 48 cases |
| candidates ranked vs timed | 40 ranked, **5 timed** |
| ranking cost | 7 ms, compiles everything, times nothing |

The honest reading: at the median it lands within 2% of the optimum, it is
inside 2% on just over half the cases, and its worst case is 38% off. It is a
shortlist generator, not an oracle, and that is what the numbers say.

### 6. A neural network, trained end to end

`kiln/nn.py`. Forward, backward and the optimiser step all run on kernels
this project emitted — the 8×8 matmul, a tiled register transpose (**4.7–5.4×
numpy**), bias+ReLU as a two-level loop, ReLU's derivative as a fused
`step` op, MSE loss as a fused reduction, and SGD-with-momentum as a fused
map that updates velocity and weights in one pass over memory.

`tests/verify_training.py` trains a 64→128→128→32 network for 300 steps, then
runs the identical network — same initial weights, same data, same
hyperparameters — in float64 numpy and compares the loss curves step by step.

```
    step     kiln loss   float64 loss    rel gap
       0    1.80536985     1.80536996   6.03e-08
      50    0.11729146     0.11729146   1.51e-08
     299    0.02984627     0.02984953   1.09e-04

loss fell 60.5x    max relative gap 1.4e-04 over 300 steps
67.7 GFLOP/s sustained    0.33 ms/step
```

Errors in backpropagation compound. A transpose off by a row, a gradient with
the wrong sign, a bias added to the wrong axis — any of those would separate
the curves in the first few steps and never let them rejoin.

---

## Being right, and how it is checked

Different operations deserve different standards, and conflating them hides
real errors.

**Bit-exact, no tolerance.** Kernels built from add, subtract, multiply,
fused-multiply-add, max, min and square root must match the reference to the
last bit for every element. 7 kernel families × 77 configurations each:
**0 ULP**.

The reference is not `float(a) * float(b)` in Python. That computes in double
and rounds twice, and it disagrees with the hardware for reasons that have
nothing to do with the compiler. `kiln/exact.py` computes every reference
operation as an exact rational and rounds it to float32 exactly once, ties to
even — the definition of what the hardware does. The first version of these
tests reported an 86-in-1000 mismatch rate that was **entirely the
reference's fault**.

**Measured, with a stated bound.** `exp`, `recip` and `rsqrt` are
approximations by construction, so there is no correct bit pattern to demand.
The tests report the worst error in ULP: exp 1, recip 1, rsqrt 2, tanh 2,
sigmoid 3.

`tanh` did not start at 2. The obvious formula, `(e^2x − 1)/(e^2x + 1)`,
subtracts two nearly equal numbers near zero and measured **254 ULP** — and
the test only caught it intermittently, because it seeded its random inputs
from Python's `hash()`, which is randomised per process. A test whose inputs
move is not a test. Both were fixed: the seeds are now derived with `crc32`,
and `tanh` is a primitive lowered to a minimax polynomial for |x| < 0.55 and
the exponential form beyond, blended with a compare-and-select because four
lanes in one register can disagree about which branch they want.

The accurate version evaluates both branches for every lane, so it costs
**1.5×** the fast one. `ir.tanh_fast` keeps the old formula for anyone who
wants that trade, with its 254 ULP stated rather than discovered later.

`gelu` reports a different metric on purpose. Its ULP number is enormous
(8.7 × 10⁸) and meaningless: the tanh form computes `1 + tanh(z)`, and where
`tanh(z) → −1` that subtraction destroys every digit — but the output there is
about 1e-7 against a function whose range is 8. Scored against the function's
scale, the error is **8.2e-08**. The cancellation is in the formula
transformers use, not in the compiler, and both numbers are printed.

**Backward error, against the exact sum.** Reductions are scored against
`math.fsum` with the bound any correct float32 summation must satisfy.
Comparing them against some *other* arbitrary summation order would measure
nothing: KILN's tree reduction is measured at 12–337× more accurate than a
sequential float32 loop.

### The reduction trade-off, published rather than buried

A plain running sum drifts, and the drift is a random walk in how many terms
each accumulator lane sees. Compensated (Kahan) summation stops it but costs
three extra vector instructions per element — about 3× on a kernel that is
already latency-bound. KILN turns compensation on once a lane would sum more
than 4,096 terms. Both halves of the trade are measured in
`bench/reduction_tradeoff.py`:

| n | plain error | plain | kahan error | kahan | numpy error | numpy |
|---|---|---|---|---|---|---|
| 65 K | 1.71e-07 | 6.8 µs | 9.60e-08 | 13.4 µs | 6.99e-09 | 24.7 µs |
| 1 M | 4.33e-06 | 120.2 µs | **4.43e-08** | **204.3** µs | 4.50e-08 | 575.9 µs |
| 16 M | 2.77e-04 | 2153.4 µs | **6.03e-08** | **3303.3** µs | 6.03e-08 | 11169.9 µs |

At 16 M elements the compensated reduction matches numpy's accuracy to three
significant figures while running **3.1× faster**.

---

## Where it wins, and where it does not

Measured against numpy on the same buffers, best-of-many-runs both sides, and
against numpy in *two* forms — the idiomatic one and the faster one an expert
would write with preallocated `out=` arrays.

| kernel | 16 M elements | vs numpy | vs numpy+`out=` |
|---|---|---|---|
| `(a*b+c*d)*(a-d) + (b*c-a*a)` | 5.4 ms | **7.91×** | 3.89× |
| `1/(1+exp(-x))` | 9.5 ms | 5.42× | 3.46× |
| `sum((a-b)²)` | 3.3 ms | 5.11× | 2.18× |
| `(a*b+c)*a - b` | 4.3 ms | 3.24× | 2.29× |
| softmax | 17.7 ms | 2.77× | 2.02× |
| `a*2.5 + b` | 3.3 ms | 2.75× | 1.23× |
| gelu | 23.5 ms | 2.43× | 1.39× |
| layernorm | 11.2 ms | 2.07× | 1.19× |

The pattern is the whole thesis: the more operators in the expression, the
bigger the win, because that is exactly how many memory passes fusion
removes. `axpy` has two operators and wins least. The nine-operator
expression wins most.

**Where numpy wins.** Matrix multiply, by roughly 25×, via AMX. That is not
close and it is not going to be.

**Accuracy against a float64 evaluation**, across all 32 benchmark
configurations: KILN's worst is 4.2e-07, numpy's is 1.6e-07, and KILN is at
least as accurate as numpy in 19 of 32 cases. The two engines land on
different bits because KILN's fused multiply-adds round once where numpy
rounds twice — "differs from numpy" and "is wrong" are different claims and
are measured separately here.

---

## Layout

```
kiln/
  isa.py          79 ARM64 instruction encoders, each carrying its own asm text
  asm.py          labels, branch fixups, printable listings
  jit.py          mmap RW -> write -> mprotect RX -> flush I-cache -> call
  exact.py        float32 arithmetic done exactly, for the reference
  ir.py           the expression DAG, hash-consed; map and reduce stages
  lower.py        liveness, unrolling, register allocation, code emission
  vecexp.py       exp() for four lanes in 21 instructions
  gemm.py         8x8 register-blocked matmul + cache blocking
  transpose.py    tiled 4x4 register transpose
  nn.py           layers, backprop and SGD, all on emitted code
  tune.py         the schedule search and its learned cost model
  runtime.py      buffers, the pointer table, timing

tour.py           a guided walkthrough that runs the compiler in front of you
tests/            verify_isa, verify_listing, verify_kernels, verify_exp,
                  verify_training
bench/            vs_numpy, roofline, reduction_tradeoff, autotune_study
tools/fit_exp.py  Remez exchange for the exp polynomial
run_all.py        runs all of it
```

`numpy` appears only in `tests/` and `bench/`, as the thing being measured
against. `tests/verify_listing.py` walks every module under `kiln/` with the
`ast` module and asserts that nothing outside the standard library is
imported.

---

## Running it

```
git clone https://github.com/GreenAiSolution/kiln && cd kiln
python3 tour.py               # a guided walkthrough, 8 stops, ~40 seconds
python3 tour.py 5             # just one stop
python3 run_all.py            # everything, about 3-4 minutes
python3 run_all.py --quick    # verification only, about 2 minutes
```

**Requirements.** macOS on Apple Silicon — the JIT path is
`mmap` → `mprotect` → `sys_icache_invalidate`, and the instruction encoder is
ARM64. `clang` and `otool` from the Xcode Command Line Tools, for the
differential tests. Python 3.9+. `numpy` only for the benchmark comparisons and
the training reference; `python3 run_all.py --quick` skips everything that
needs it except `verify_training.py`.

### Reproducing the numbers

Every figure in this README comes out of `run_all.py`. What to expect:

| suite | time | what it must print |
|---|---|---|
| `verify_isa` | 1 s | `MISMATCHES : 0` over 491 instructions |
| `verify_listing` | 3 s | 0 mismatches over 159 kernels; 0 non-stdlib imports |
| `verify_kernels` | 130 s | `0` in the ULP column for all 7 EXACT families |
| `verify_exp` | 1 s | `max 1 ULP` on every sweep |
| `verify_training` | 1 s | `max relative gap` below 5e-3 |
| `roofline` | 12 s | FMA peak near 4.00/cycle; matmul >90% at 128³ |
| `vs_numpy` | 35 s | median speedup above 2× |
| `reduction_tradeoff` | 10 s | compensated error at 16 M within 2× of numpy's |
| `autotune_study` | 40 s | guided/best median under 1.05× |

**If a number differs from this README**, that is expected and fine — these
are measurements of a specific machine. The correctness suites
(`verify_*`) are the ones that must pass identically everywhere; they contain
no timing at all. The benchmark suites will move with your chip, your clock,
your thermal state and your numpy build.

Two things that will make timings read low, both of which cost me real time:

- **A cold core.** Apple Silicon ramps frequency on demand, so the first thing
  timed in a process runs slow. `kiln.runtime.spin_up()` handles this; if you
  write your own benchmark, call it first.
- **Anything else running.** These are single-core measurements taken as a
  best-of-many. Other load only ever makes them worse.

`results/` holds the transcripts from the run these numbers were taken from,
so you can diff yours against mine.

## How this relates to work that already exists

This is not a new idea, and pretending otherwise would be the fastest way to
lose an informed reader.

**Halide, TVM, Triton, XLA and tinygrad** all do this, at industrial scale,
with far more capability: multiple backends, GPUs, autoscheduling searched
over spaces vastly larger than 40 candidates, and years of production use.
Anyone who needs a tensor compiler should use one of those, not this.

What is different here is narrow and deliberate:

- **Every one of them rests on LLVM** (or on a vendor assembler, or on
  CUDA/Metal) for the last step — turning an instruction into bytes. KILN does
  that step itself. That is the part this project exists to show.
- **The verification is unusual.** Emitting machine code is easy to get subtly
  wrong and hard to notice. Making every encoder carry its own assembly text,
  so the whole instruction set can be differential-tested against the platform
  assembler, is a cheap technique that more projects in this space could use.
  It cost about 150 lines and caught a real bug immediately.
- **It is small enough to read.** ~3,200 lines with no dependencies means the
  path from `a*b + c` to a 32-bit instruction word is followable end to end,
  which is not true of any of the above.

If you want the same idea done properly and at scale, read tinygrad — it is
the closest in spirit and about as legible as a real one gets.

## Known limits

- Apple Silicon and ARM64 only. The instruction encoder is architecture
  specific; the compiler passes above it are not.
- float32 only.
- `gemm` requires M%8, N%8, K%4 and matrix strides under ~2,300 columns for
  immediate addressing; `matmul()` pads awkward shapes at the cost of a copy.
- Single core. Nothing here threads.
- The cost model's worst case is 1.62× off the optimum. It shortlists; it
  does not decide.
- MAP_JIT is not used — that path needs a code-signing entitlement Python does
  not have here, so KILN maps read/write and flips to read/execute instead.
