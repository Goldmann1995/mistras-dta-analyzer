"""Unified AE feature-extraction comparison framework.

Compares 6 feature-extraction methods (M1..M6) for clustering unlabeled
acoustic-emission (AE) hits from composites (CFRP/GFRP), under identical
data / preprocessing / clusterers / metrics. See ``README.md`` and
``run_compare.py``.
"""

__all__ = ["data", "encoders", "clustering", "metrics", "viz", "pipeline"]
