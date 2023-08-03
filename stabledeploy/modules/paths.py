import os
import sys
from .paths_internal import models_path, script_path, data_path, extensions_dir, extensions_builtin_dir  # noqa: F401

from . import safe  # noqa: F401


BLIP_path = os.path.join(script_path, 'repositories/BLIP')

path_dirs = [
    (os.path.join(BLIP_path, '../CodeFormer'), 'inference_codeformer.py', 'CodeFormer', []),
    (os.path.join(BLIP_path, '../BLIP'), 'models/blip.py', 'BLIP', []),
    # (os.path.join(BLIP_path, '../k-diffusion'), 'k_diffusion/sampling.py', 'k_diffusion', ["atstart"]),
]

print(f"[tangjicheng] script_path: {script_path}")

paths = {}

for d, must_exist, what, options in path_dirs:
    must_exist_path = os.path.abspath(os.path.join(script_path, d, must_exist))
    if not os.path.exists(must_exist_path):
        print(f"Warning: {what} not found at path {must_exist_path}", file=sys.stderr)
    else:
        d = os.path.abspath(d)
        paths[what] = d
