"""
Tölvera LLM Integration - Clean, modular implementation.

Main exports for easy importing with focus on slime + flock interactions.
"""

# Core agent functionality
from .agent import (
    SketchAgent
)

# Models (unchanged - they're already clean)
from .models import (
    MultiBehaviorSketchConfig,
    SketchGenerationResponse,
    SketchModificationResponse,
    SketchTemplates,
    BehaviorInstance,
    BehaviorType,
    FlockBehaviorConfig,
    SlimeBehaviorConfig,
    ColorPalette,
    RenderConfig
)

# Code generation
from .code_generation import JinjaCodeGenerator, create_code_generator

# Utilities
from .utils import get_best_available_model, check_ollama_connection, OllamaModelManager

__all__ = [
    # Core functionality
    "SketchAgent",
    
    # Models
    "MultiBehaviorSketchConfig",
    "SketchGenerationResponse",
    "SketchModificationResponse", 
    "SketchTemplates",
    "BehaviorInstance",
    "BehaviorType",
    "FlockBehaviorConfig",
    "SlimeBehaviorConfig",
    "ColorPalette",
    "RenderConfig",
    
    # Code generation
    "JinjaCodeGenerator",
    "create_code_generator",
    
    # Utils
    "get_best_available_model",
    "check_ollama_connection",
    "OllamaModelManager"
]