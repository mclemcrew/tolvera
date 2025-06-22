# src/tolvera/llm/compositional/__init__.py
"""
Compositional MoE system for Tölvera.
Implements the architectural blueprint for natural language to code generation.

This module provides a Mixture-of-Experts system that can generate complete
Tölvera Python scripts from natural language descriptions.

Example usage:
    from tolvera.llm.compositional import CodeGenerationOrchestrator
    
    orchestrator = CodeGenerationOrchestrator()
    script = await orchestrator.generate_script("move a blue pixel from left to right")
    
    # Save the generated script
    with open("my_sketch.py", "w") as f:
        f.write(script.code)
"""

# Main orchestrator classes
from .orchestrator import CodeGenerationOrchestrator, SimpleOrchestrator, create_orchestrator

# Expert agents (for advanced usage)
from .agents import (
    ConductorAgent,
    ParticleCreationAgent, 
    ColorPaletteAgent,
    MotionDynamicsAgent,
    PhysicsAgent,
    CompositionAgent,
    BaseAgent,
    create_all_agents,
    check_agent_models,
    AGENT_SPECIALIZATIONS
)

# Tool definitions and schemas
from .tools import (
    # Pydantic schemas
    CreateParticlesSchema,
    SetParticleVelocitySchema,
    SetSpeciesVelocitySchema,
    SetParticleColorSchema,
    SetSpeciesColorSchema,
    FillBackgroundSchema,
    SetGlobalStateSchema,
    ApplyFlockBehaviorSchema,
    ApplyNoiseFieldSchema,
    ApplyVaryingSpeedsSchema,
    
    # Communication schemas
    ToolCall,
    TaskResult,
    TaskPlan,
    GeneratedScript,
    
    # Utilities
    STANDARD_COLORS,
    TOOL_REGISTRY,
    get_color_by_name,
    validate_tool_call,
    normalize_position,
    
    # Templates
    generate_tolvera_imports,
    generate_main_function_template,
    
    # Tool descriptions for agents
    PARTICLE_AGENT_TOOLS,
    COLOR_AGENT_TOOLS,
    MOTION_AGENT_TOOLS,
    PHYSICS_AGENT_TOOLS
)

# Utility functions
from .utils import (
    # Model checking
    check_ollama_connection,
    get_available_models,
    check_required_models,
    get_best_available_models,
    print_model_status,
    
    # File utilities
    ensure_directory,
    save_generated_script,
    create_test_directory,
    sanitize_filename,
    
    # Validation
    validate_script_syntax,
    check_tolvera_imports,
    validate_generated_script,
    
    # Logging
    setup_logging,
    log_agent_performance,
    
    # Examples and templates
    EXAMPLE_REQUESTS,
    TEST_REQUESTS,
    
    # Performance monitoring
    PerformanceMonitor,
    
    # Integration helpers
    create_integration_example,
    check_dependencies,
    print_dependency_status
)

# Version info
__version__ = "0.1.0"
__author__ = "Tölvera MoE System"
__description__ = "Mixture-of-Experts system for generating Tölvera sketches from natural language"

# Main exports for public API
__all__ = [
    # Primary classes users will interact with
    "CodeGenerationOrchestrator",
    "SimpleOrchestrator", 
    "create_orchestrator",
    
    # Core data structures
    "GeneratedScript",
    "ToolCall",
    "TaskResult", 
    "TaskPlan",
    
    # Utility functions for setup and validation
    "print_model_status",
    "print_dependency_status", 
    "check_dependencies",
    "save_generated_script",
    "validate_generated_script",
    
    # Performance monitoring
    "PerformanceMonitor",
    
    # Example data for testing
    "EXAMPLE_REQUESTS",
    "TEST_REQUESTS",
    
    # Expert agents (for advanced usage)
    "ConductorAgent",
    "ParticleCreationAgent",
    "ColorPaletteAgent", 
    "MotionDynamicsAgent",
    "PhysicsAgent",
    "CompositionAgent",
    
    # Color utilities
    "STANDARD_COLORS",
    "get_color_by_name",
]

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def quick_generate(request: str, save_file: str = None) -> GeneratedScript:
    """
    Convenience function to quickly generate a script from a request.
    
    Args:
        request: Natural language request
        save_file: Optional filename to save the script
        
    Returns:
        GeneratedScript object
        
    Example:
        script = await quick_generate("move a blue pixel from left to right", "blue_pixel.py")
    """
    orchestrator = CodeGenerationOrchestrator()
    script = await orchestrator.generate_script(request)
    
    if save_file:
        save_generated_script(script.code, save_file)
    
    return script

def check_system_ready() -> bool:
    """
    Check if the MoE system is ready to use.
    
    Returns:
        True if all dependencies are satisfied
    """
    return print_dependency_status()

def get_example_requests() -> list:
    """Get a list of example requests for testing."""
    return EXAMPLE_REQUESTS.copy()

def get_system_info() -> dict:
    """Get comprehensive system information."""
    return {
        "version": __version__,
        "dependencies": check_dependencies(),
        "models": check_required_models(),
        "best_models": get_best_available_models(),
        "example_requests": len(EXAMPLE_REQUESTS),
        "available_agents": list(AGENT_SPECIALIZATIONS.keys())
    }

# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

def _check_environment():
    """Check the environment on module import."""
    import logging
    logger = logging.getLogger(__name__)
    
    # Check basic dependencies
    deps = check_dependencies()
    
    if not deps.get("pydantic_ai", False):
        logger.warning("pydantic-ai not found. Install with: pip install pydantic-ai")
    
    if not deps.get("ollama_connection", False):
        logger.warning("Ollama not accessible. Start with: ollama serve")
    
    if not deps.get("required_models", False):
        logger.warning("Required models not found. Install with: ollama pull llama3.2:3b")

# Run environment check on import (but don't fail if there are issues)
try:
    _check_environment()
except Exception:
    pass  # Don't fail module import due to environment checks