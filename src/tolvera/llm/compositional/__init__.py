# src/tolvera/llm/compositional/__init__.py
"""
Compositional MoE system for Tölvera.
Implements the architectural blueprint for natural language to code generation.

This module provides a Mixture-of-Experts system that can generate complete
Tölvera Python scripts from natural language descriptions.

Example usage:
    from tolvera.llm.compositional import RobustCodeGenerationOrchestrator
    
    orchestrator = RobustCodeGenerationOrchestrator()
    script = await orchestrator.generate_script("move a blue pixel from left to right")
    
    # Save the generated script
    with open("my_sketch.py", "w") as f:
        f.write(script.code)
"""

# =============================================================================
# MAIN ORCHESTRATOR CLASSES (ROBUST VERSIONS)
# =============================================================================

# Primary robust orchestrator (recommended)
from .orchestrator import RobustCodeGenerationOrchestrator, create_robust_orchestrator


# =============================================================================
# EXPERT AGENTS (ROBUST VERSIONS RECOMMENDED)
# =============================================================================

# Robust expert agents (recommended)
from .agents import (
    RobustConductorAgent,
    RobustParticleCreationAgent, 
    RobustColorPaletteAgent,
    RobustMotionDynamicsAgent,
    RobustPhysicsAgent,
    RobustCompositionAgent,
    RobustBaseAgent,
    create_robust_agents,
)

# =============================================================================
# TOOL DEFINITIONS AND SCHEMAS
# =============================================================================

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
    
    # Templates and code generation
    generate_tolvera_imports,
    generate_main_function_template,
    tool_calls_to_python_code,
    
    # Tool descriptions for agents
    PARTICLE_AGENT_TOOLS,
    COLOR_AGENT_TOOLS,
    MOTION_AGENT_TOOLS,
    PHYSICS_AGENT_TOOLS
)

# =============================================================================
# JSON UTILITIES (NEW)
# =============================================================================

from .json_utils import (
    clean_json_response,
    fix_common_json_issues,
    safe_json_parse,
    validate_tool_call_json,
    validate_task_plan_json,
    extract_json_from_text,
)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

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

# =============================================================================
# VERSION INFO AND AGENT SPECIALIZATIONS
# =============================================================================

__version__ = "0.2.0"  # Updated version for robust system
__author__ = "Tölvera MoE System"
__description__ = "Robust Mixture-of-Experts system for generating Tölvera sketches from natural language"

# Agent specializations
AGENT_SPECIALIZATIONS = {
    "conductor": "Master planner and task decomposition using Chain-of-Thought",
    "particle": "Particle creation and positioning expert",
    "color": "Color management and palette expert", 
    "motion": "Movement and velocity dynamics expert",
    "physics": "Complex physics interactions and behaviors expert",
    "composition": "Final script assembly and code generation expert"
}

# =============================================================================
# MAIN EXPORTS FOR PUBLIC API
# =============================================================================

__all__ = [
    # === PRIMARY ROBUST CLASSES (RECOMMENDED) ===
    "RobustCodeGenerationOrchestrator",  # Main robust MoE orchestrator
    "create_robust_orchestrator",        # Factory for robust orchestrator

    
    # === CORE DATA STRUCTURES ===
    "GeneratedScript",                   # Generated script container
    "ToolCall",                          # Tool call representation
    "TaskResult",                        # Agent task result
    "TaskPlan",                          # Conductor plan
    
    # === ROBUST EXPERT AGENTS (RECOMMENDED) ===
    "RobustConductorAgent",              # Robust master planner
    "RobustParticleCreationAgent",       # Robust particle creation expert
    "RobustColorPaletteAgent",           # Robust color management expert
    "RobustMotionDynamicsAgent",         # Robust movement expert
    "RobustPhysicsAgent",                # Robust physics expert
    "RobustCompositionAgent",            # Robust code assembly expert
    "create_robust_agents",              # Factory for robust agents
    
    # === UTILITY FUNCTIONS ===
    "print_model_status",                # Check available models
    "print_dependency_status",           # Check all dependencies
    "check_dependencies",                # Dependency status dict
    "save_generated_script",             # Save script to file
    "validate_generated_script",         # Validate generated code
    "tool_calls_to_python_code",         # Direct code generation
    
    # === JSON UTILITIES ===
    "clean_json_response",               # Clean LLM JSON responses
    "safe_json_parse",                   # Safe JSON parsing
    "validate_tool_call_json",           # Validate tool call structure
    "validate_task_plan_json",           # Validate task plan structure
    
    # === PERFORMANCE AND MONITORING ===
    "PerformanceMonitor",                # Performance tracking
    
    # === EXAMPLES AND TESTING ===
    "EXAMPLE_REQUESTS",                  # Example prompts for testing
    "TEST_REQUESTS",                     # Test prompts
    
    # === COLOR UTILITIES ===
    "STANDARD_COLORS",                   # Standard color definitions
    "get_color_by_name",                 # Get color by name
    
    # === METADATA ===
    "AGENT_SPECIALIZATIONS",             # Agent role descriptions
]

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def quick_generate(request: str, save_file: str = None, use_robust: bool = True) -> GeneratedScript:
    """
    Convenience function to quickly generate a script from a request.
    
    Args:
        request: Natural language request
        save_file: Optional filename to save the script
        use_robust: Whether to use the robust orchestrator (recommended)
        
    Returns:
        GeneratedScript object
        
    Example:
        script = await quick_generate("move a blue pixel from left to right", "blue_pixel.py")
    """
    if use_robust:
        orchestrator = RobustCodeGenerationOrchestrator()
        
    script = await orchestrator.generate_script(request)
    
    if save_file:
        save_generated_script(script.code, save_file)
    
    return script

async def robust_quick_generate(request: str, save_file: str = None) -> GeneratedScript:
    """
    Convenience function using the robust orchestrator specifically.
    
    Args:
        request: Natural language request
        save_file: Optional filename to save the script
        
    Returns:
        GeneratedScript object
    """
    return await quick_generate(request, save_file, use_robust=True)

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
        "available_agents": list(AGENT_SPECIALIZATIONS.keys()),
        "robust_system": True,
        "json_utilities": True,
        "fallback_support": True
    }

# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_recommended_orchestrator() -> RobustCodeGenerationOrchestrator:
    """Create the recommended robust orchestrator."""
    return RobustCodeGenerationOrchestrator()

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

# =============================================================================
# USAGE RECOMMENDATIONS
# =============================================================================

def print_usage_recommendations():
    """Print usage recommendations for the robust system."""
    print("""
🎯 Tölvera MoE System - Usage Recommendations
=============================================

RECOMMENDED (Robust System):
  from tolvera.llm.compositional import RobustCodeGenerationOrchestrator
  orchestrator = RobustCodeGenerationOrchestrator()

CONVENIENCE FUNCTION (Recommended):
  from tolvera.llm.compositional import robust_quick_generate
  script = await robust_quick_generate("your request", "output.py")

LEGACY (Backward Compatibility):
  from tolvera.llm.compositional import CodeGenerationOrchestrator
  orchestrator = CodeGenerationOrchestrator()

KEY IMPROVEMENTS IN ROBUST SYSTEM:
  ✅ Better JSON parsing and error handling
  ✅ Fallback mechanisms at every level
  ✅ Individual agent isolation
  ✅ Keyword-based fallback generation
  ✅ Direct code composition (no LLM for final step)
  ✅ Comprehensive error recovery

TESTING:
  Use test_robust_system.py to verify functionality
""")

# Add to __all__ for completeness
__all__.extend([
    "quick_generate",
    "robust_quick_generate", 
    "check_system_ready",
    "get_example_requests",
    "get_system_info",
    "create_recommended_orchestrator",
    "create_legacy_orchestrator",
    "print_usage_recommendations"
])