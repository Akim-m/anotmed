"""Test bootstrap.

Puts the project root on sys.path (so `anotmed` and `examples` import from
source without an install) and forces the CPU stub backend with a throwaway
storage dir, so no test can accidentally load model weights or touch real data.
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("ANOTMED_BACKEND", "stub")
os.environ.setdefault("ANOTMED_STORAGE", tempfile.mkdtemp(prefix="anotmed_test_"))
