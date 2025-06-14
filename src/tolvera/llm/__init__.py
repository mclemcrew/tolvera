### This is a messy import with just a full dump and needs to be cleaned up!
try:
    from .models import (
        MultiBehaviorSketchConfig,
        BehaviorInstance,
        BehaviorType,
        FlockBehaviorConfig,
        SlimeBehaviorConfig,
        ParticleLifeConfig,
        ColorPalette,
        RenderConfig,
        ParticleShape,
        BackgroundBehavior,
        SketchTemplates,
        SketchGenerationResponse,
        SketchModificationResponse
    )

    from .agent import (
        TolveraSketchAgent,
        generate_tolvera_sketch,
        modify_tolvera_sketch
    )
except ImportError:
    # I was getting errors so this is more of a problem as well
    try:
        from models import (
            MultiBehaviorSketchConfig,
            BehaviorInstance,
            BehaviorType,
            FlockBehaviorConfig,
            SlimeBehaviorConfig,
            ParticleLifeConfig,
            ColorPalette,
            RenderConfig,
            ParticleShape,
            BackgroundBehavior,
            SketchTemplates,
            SketchGenerationResponse,
            SketchModificationResponse
        )

        from agent import (
            TolveraSketchAgent,
            generate_tolvera_sketch,
            modify_tolvera_sketch
        )
    except ImportError as e:
        print(f"Warning: Could not import tolvera.llm components: {e}")
        # Define empty module for graceful degradation
        class _MockClass: pass
        MultiBehaviorSketchConfig = _MockClass
        BehaviorInstance = _MockClass
        BehaviorType = _MockClass
        TolveraSketchAgent = _MockClass

# Main exports
__all__ = [
    # Core models
    "MultiBehaviorSketchConfig",
    "BehaviorInstance", 
    "BehaviorType",
    "FlockBehaviorConfig",
    "SlimeBehaviorConfig",
    "ParticleLifeConfig",
    "ColorPalette",
    "RenderConfig",
    "ParticleShape",
    "BackgroundBehavior",
    "SketchTemplates",
    
    # Response models
    "SketchGenerationResponse",
    "SketchModificationResponse",
    
    # Agent classes and functions
    "TolveraSketchAgent",
    "generate_tolvera_sketch",
    "modify_tolvera_sketch",
]

# Only export what actually exists
__all__ = [name for name in __all__ if globals().get(name) is not None]

# Version info
__version__ = "0.1.0"
__author__ = "Your Name"
__description__ = "PydanticAI-powered multi-behavior Tölvera sketch generation"