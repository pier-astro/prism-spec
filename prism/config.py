"""
Package-level configuration flags for prism-spec.

Flags
-----
FORCE_OP_MISMATCH : bool
    Allow arithmetic operations between ``Cube`` / ``Image`` objects that
    have mismatched shapes or WCS.  ``False`` by default; enable only for
    exploratory analysis.
"""

# Allow mismatched-shape / WCS arithmetic on Cube / Image objects.
FORCE_OP_MISMATCH = False

