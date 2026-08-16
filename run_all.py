"""
Run everything KILN claims, in order, and print the evidence.

    python3 run_all.py            # verification + benchmarks  (~6 min)
    python3 run_all.py --quick    # verification only          (~2 min)

Every number this prints is measured on the machine it is run on. Nothing is
copied from a previous run and nothing is hard-coded.
"""

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

VERIFY = [
    ("Instruction encoder vs Apple's assembler", "tests/verify_isa.py", []),
    ("Whole kernels + dependency audit", "tests/verify_listing.py", []),
    ("Numerics of every generated kernel", "tests/verify_kernels.py", []),
    ("Vectorised exp, on hardware", "tests/verify_exp.py", []),
    ("Neural network vs a float64 reference", "tests/verify_training.py", []),
]

BENCH = [
    ("Machine ceilings and matmul roofline", "bench/roofline.py", []),
    ("Fused kernels vs numpy", "bench/vs_numpy.py",
     ["--seconds", "0.08", "--json", "results/vs_numpy.json"]),
    ("Reduction accuracy vs speed", "bench/reduction_tradeoff.py", []),
    ("Cost model, leave-one-kernel-out", "bench/autotune_study.py",
     ["--seconds", "0.012", "--shortlist", "5",
      "--json", "results/autotune.json"]),
]


def run(title, script, args):
    print()
    print("#" * 78)
    print(f"# {title}")
    print(f"# {script} {' '.join(args)}")
    print("#" * 78)
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, os.path.join(HERE, script)] + args,
                       cwd=HERE)
    dt = time.perf_counter() - t0
    print(f"[{title}: {'ok' if r.returncode == 0 else 'FAILED'} in {dt:.1f}s]")
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="verification only, skip the benchmarks")
    args = ap.parse_args()

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    plan = list(VERIFY) + ([] if args.quick else list(BENCH))

    t0 = time.perf_counter()
    results = [(title, run(title, s, a)) for title, s, a in plan]
    dt = time.perf_counter() - t0

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for title, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {title}")
    bad = [t for t, ok in results if not ok]
    print(f"\n{len(results) - len(bad)}/{len(results)} passed in {dt:.0f}s")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
