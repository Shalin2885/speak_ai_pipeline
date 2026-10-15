"""
pipeline/__init__.py
Speak AI Pipeline — public package surface.
"""

from .config import PipelineConfig
from .orchestrator import Orchestrator

__all__ = ["PipelineConfig", "Orchestrator"]
__version__ = "0.1.0"
