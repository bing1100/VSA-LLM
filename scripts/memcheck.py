"""User-space RAM pattern check: write address-dependent patterns into a large array and re-read.

A healthy machine reports 0 bad words. Usage: python scripts/memcheck.py [GiB] [passes]
This is a quick screen, not a replacement for memtest86+ (which tests all memory outside the OS).
"""
import sys
import time

import numpy as np

gib = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
passes = int(sys.argv[2]) if len(sys.argv) > 2 else 4
n = int(gib * 2**30) // 8
a = np.empty(n, dtype=np.uint64)
index = np.arange(n, dtype=np.uint64)
seeds = [0x5555555555555555, 0xAAAAAAAAAAAAAAAA, 0x0123456789ABCDEF, 0xFEDCBA9876543210]
total = 0
for p in range(passes):
    seed = np.uint64(seeds[p % len(seeds)])
    start = time.time()
    a[:] = seed
    a ^= index
    for check in range(3):
        bad = int(((a ^ index) != seed).sum())
        total += bad
        print(f"pass {p} check {check}: {bad} bad words", flush=True)
    print(f"pass {p} done in {time.time() - start:.0f}s", flush=True)
print(f"TOTAL bad words: {total} ({'FAIL' if total else 'OK'})")
sys.exit(1 if total else 0)
