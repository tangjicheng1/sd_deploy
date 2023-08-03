import os
import sys
from .paths_internal import models_path, script_path, data_path, extensions_dir, extensions_builtin_dir  # noqa: F401

from . import safe  # noqa: F401


sys.path.insert(0, script_path)

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
        if "atstart" in options:
            sys.path.insert(0, d)
        else:
            sys.path.append(d)
        paths[what] = d


class Prioritize:
    def __init__(self, name):
        self.name = name
        self.path = None

    def __enter__(self):
        self.path = sys.path.copy()
        sys.path = [paths[self.name]] + sys.path

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.path = self.path
        self.path = None
