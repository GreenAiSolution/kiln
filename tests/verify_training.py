"""
Train the same network twice and see whether the curves agree.

Run 1 is KILN: every multiply, activation, gradient and weight update is
machine code this project generated, running on buffers this project
allocated.

Run 2 is the same network written in plain float64 numpy - same initial
weights read straight out of KILN's buffers, same data, same learning rate,
same momentum, same number of steps.

If any kernel were wrong - a transpose off by a row, a gradient with the
wrong sign, a bias added to the wrong axis - the two loss curves would
separate immediately and stay separated. Agreement across hundreds of steps
is a strong statement, because errors in backpropagation compound.

The residual gap that does remain is float32 rounding, and it is reported
rather than tolerated silently.
"""

import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                            # noqa: E402

from kiln import ir, jit, nn                                  # noqa: E402
from kiln.runtime import bench_batched                        # noqa: E402

BATCH = 128
DIMS = [64, 128, 128, 32]
STEPS = 300
LR = 0.05
MOM = 0.9


def make_task(seed=7):
    """A fixed random teacher network. The student has to reproduce it -
    a real, non-trivial, exactly reproducible regression problem."""
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 1.0, (DIMS[0], 96)).astype(np.float32)
    W2 = rng.normal(0, 1.0, (96, DIMS[-1])).astype(np.float32)

    def gen(n):
        X = rng.normal(0, 1, (n, DIMS[0])).astype(np.float32)
        H = np.maximum(X @ W1 / math.sqrt(DIMS[0]), 0)
        T = np.tanh(H @ W2 / math.sqrt(96)).astype(np.float32)
        return X, T

    return gen


def to_buf(arr):
    b = jit.Buf(arr.size)
    b.frombytes(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
    return b


def numpy_reference(weights, X, T, steps, lr, mom):
    """The same network, in float64, with nothing clever."""
    Ws = [w.astype(np.float64).copy() for w, _ in weights]
    bs = [b.astype(np.float64).copy() for _, b in weights]
    vW = [np.zeros_like(w) for w in Ws]
    vb = [np.zeros_like(b) for b in bs]
    X64, T64 = X.astype(np.float64), T.astype(np.float64)
    L = len(Ws)
    curve = []
    for _ in range(steps):
        acts = [X64]
        h = X64
        for i in range(L):
            z = h @ Ws[i] + bs[i]
            h = np.maximum(z, 0) if i < L - 1 else z
            acts.append(h)
        Y = acts[-1]
        d = Y - T64
        loss = float((d * d).mean())
        curve.append(loss)

        g = 2.0 * d / d.size
        for i in range(L - 1, -1, -1):
            if i < L - 1:
                g = g * (acts[i + 1] > 0)
            dW = acts[i].T @ g
            db = g.sum(axis=0)
            gnext = g @ Ws[i].T
            vW[i] = mom * vW[i] - lr * dW
            vb[i] = mom * vb[i] - lr * db
            Ws[i] += vW[i]
            bs[i] += vb[i]
            g = gnext
    return curve


def main():
    gen = make_task()
    X, T = gen(BATCH)
    bX, bT = to_buf(X), to_buf(T)

    net = nn.MLP(BATCH, DIMS, seed=3)
    weights = []
    for l in net.layers:
        W = np.asarray(l.W.tolist(), dtype=np.float32).reshape(l.din, l.dout)
        b = np.asarray(l.b.tolist(), dtype=np.float32)
        weights.append((W, b))

    print("KILN neural network - trained entirely on emitted machine code")
    print("=" * 74)
    print(f"architecture     : {' -> '.join(map(str, DIMS))}   batch {BATCH}")
    print(f"parameters       : {net.n_params():,}")
    print(f"steps            : {STEPS}   lr {LR}   momentum {MOM}")

    curve = []
    t0 = time.perf_counter()
    for _ in range(STEPS):
        Y = net.forward(bX)
        curve.append(net.loss(Y, bT))
        net.backward(Y, bT)
        net.step(lr=LR, momentum=MOM)
    train_s = time.perf_counter() - t0

    ref = numpy_reference(weights, X, T, STEPS, LR, MOM)

    print()
    print(f"  {'step':>6} {'kiln loss':>13} {'float64 loss':>14} "
          f"{'rel gap':>10}")
    for i in [0, 1, 2, 5, 10, 25, 50, 100, 200, STEPS - 1]:
        if i >= len(curve):
            continue
        gap = abs(curve[i] - ref[i]) / max(abs(ref[i]), 1e-12)
        print(f"  {i:>6} {curve[i]:>13.8f} {ref[i]:>14.8f} {gap:>10.2e}")

    gaps = [abs(a - b) / max(abs(b), 1e-12) for a, b in zip(curve, ref)]
    print()
    print(f"loss at step 0   : {curve[0]:.6f}")
    print(f"loss at step {STEPS - 1:<3} : {curve[-1]:.6f}   "
          f"({curve[0] / curve[-1]:.1f}x lower)")
    print(f"float64 reference: {ref[-1]:.6f}")
    print(f"max relative gap : {max(gaps):.3e} over all {STEPS} steps")
    print(f"gap at final step: {gaps[-1]:.3e}")

    monotone = sum(1 for i in range(1, len(curve)) if curve[i] <= curve[i - 1])
    print(f"steps that improved: {monotone}/{STEPS - 1}")

    fps = net.flops_per_step()
    print()
    print(f"training time    : {train_s * 1e3:.0f} ms for {STEPS} steps "
          f"({train_s / STEPS * 1e3:.2f} ms/step)")
    print(f"arithmetic       : {fps / 1e6:.1f} MFLOP per step, "
          f"{fps * STEPS / train_s / 1e9:.1f} GFLOP/s sustained")

    import json
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "results"), exist_ok=True)
    out = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "results", "training_curve.json")
    with open(out, "w") as f:
        json.dump({"kiln": curve, "float64": ref, "steps": STEPS,
                   "dims": DIMS, "batch": BATCH, "lr": LR, "momentum": MOM,
                   "max_rel_gap": max(gaps),
                   "ms_per_step": train_s / STEPS * 1e3}, f)
    print(f"curve written to {out}")

    ok = max(gaps) < 5e-3 and curve[-1] < curve[0] * 0.5
    print()
    if ok:
        print("PASS - the float32 network tracks the float64 reference and "
              "the loss came down.")
    else:
        print("FAIL - curves diverged or the loss did not fall.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
