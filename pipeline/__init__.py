"""
pipeline/__init__.py
Speak AI Pipeline — public package surface.
"""

from .config import PipelineConfig


def __getattr__(name: str):
    if name == "Orchestrator":
        from .orchestrator import Orchestrator
        return Orchestrator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["PipelineConfig", "Orchestrator"]
__version__ = "0.1.0"

