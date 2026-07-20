"""Test-suite fixtures/guards.

A developer's shell may export ANOTMED_MODALITY; without this, that would leak the
active modality profile into every Config() the suite constructs and change
profile-backed defaults. Pin it to "" (generic) unless a test sets it explicitly.
"""
import os

os.environ.setdefault("ANOTMED_MODALITY", "")
