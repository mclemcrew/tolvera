"""
Enhanced utilities for Tölvera LLM integration.

Model management and helper functions with focus on reliability and ease of use.
"""

from .model_management import (
    OllamaModelManager,
    get_available_models,
    get_best_available_model,
    check_ollama_connection
)

# Add the new API inspector functions
try:
    from .api_inspector import (
        TolveraAPIInspector,
        create_api_inspector,
        quick_slime_check
    )
except ImportError:
    # If api_inspector module doesn't exist yet, create placeholder functions
    def create_api_inspector(*args, **kwargs):
        raise ImportError("API Inspector not available - create api_inspector.py module")
    
    def quick_slime_check():
        raise ImportError("API Inspector not available - create api_inspector.py module")

__all__ = [
    "OllamaModelManager",
    "get_available_models", 
    "get_best_available_model",
    "check_ollama_connection",
    "TolveraAPIInspector",
    "create_api_inspector",
    "quick_slime_check"
]