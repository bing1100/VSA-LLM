"""Patch the causal-conv1d 1.7.0 sdist's setup.py for a local sm_86-only build with at most 4 compile threads.

    python experiments/e9-retrofit/env/patch_causal_conv1d.py <unpacked causal_conv1d-1.7.0>

The sdist's gencode list (sm_75/80/87/90/100/120) has no sm_86 entry (sm_80 code would run on the RTX 3090) and
nvcc runs with `--threads 4` per job; this keeps sm_86 only and one thread per nvcc job (MAX_JOBS=4 → 4 threads).
"""
import sys
from pathlib import Path

path = Path(sys.argv[1]) / "setup.py"
text = path.read_text()
start = text.index('        cc_flag.append("-gencode")\n        cc_flag.append("arch=compute_75,code=sm_75")')
end = text.index("    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI")
text = text[:start] + ('        # local build (WP-Qwen35): RTX 3090 only\n'
                       '        cc_flag.append("-gencode")\n'
                       '        cc_flag.append("arch=compute_86,code=sm_86")\n\n') + text[end:]
text = text.replace('return nvcc_extra_args + ["--threads", "4"]', 'return nvcc_extra_args + ["--threads", "1"]')
assert "compute_75" not in text and '"--threads", "1"' in text
path.write_text(text)
print("patched", path)
