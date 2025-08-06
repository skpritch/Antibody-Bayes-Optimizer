"""
Antibody Bayesian‑Optimisation framework
=======================================

Convenience::

    from antibody_bo.pipeline import AntibodyBOPipeline, PipelineConfig
"""
from importlib import metadata as _meta

__all__ = [
    "embeddings",
    "models",
    "acquisition",
    "optimisation",
    "developability",
    "utils",
    "pipeline",
]
__version__ = _meta.version(__name__, default="0.2.0")
