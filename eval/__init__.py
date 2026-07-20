"""anotmed validation harness (tier 2: slow, real, run manually or nightly).

Separate from `tests/` (tier 1: fast, mocked, CPU). This package scores a backend
against ground truth and enforces absolute floors — the safety gate that must
pass before any real clinical-adjacent use. See PLAN.md Phase 3.
"""
