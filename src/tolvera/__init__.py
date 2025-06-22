# tolvera/__init__.py

from fire import Fire as run

from .context import TolveraContext
from .particles import *
from .patches import *
from .pixels import *
from .state import StateDict
from .utils import *
from .vera import Vera

# CORRECTED: Import the correct class name
from .llm import CodeGenerationOrchestrator  # Changed from SketchOrchestrator

from .tolvera_ import Tolvera
from .rec import VideoRecorder
from .dualsense import DualSense

__all__ = [
    "run",
    "TolveraContext",
    "Particle",
    "Particles",
    "Pixel",
    "Pixels",
    "StateDict",
    "Vera",
    "CodeGenerationOrchestrator",  # Changed from SketchOrchestrator
    "Tolvera",
    "VideoRecorder",
    "DualSense",
]