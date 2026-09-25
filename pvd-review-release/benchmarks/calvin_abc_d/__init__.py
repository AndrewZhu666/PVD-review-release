"""Portable CALVIN ABC-D benchmark entry points.

The package deliberately keeps CALVIN/RLinf imports behind the command-line
entry points.  Reviewers can therefore inspect and validate the protocol in a
CPU-only environment, while a real run receives every simulator/model path
explicitly from the caller.
"""

__all__ = ["common", "environment", "evaluate", "train"]
