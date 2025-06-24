# src/tolvera/llm/compositional/__init__.py
"""
Compositional MoE system for Tölvera.
Implements the architectural blueprint for natural language to code generation.

This module provides a Mixture-of-Experts system that can generate complete
Tölvera Python scripts from natural language descriptions with improved robustness,
better JSON handling, and more organic particle behaviors.

Example usage:
    from tolvera.llm.compositional import RobustCodeGenerationOrchestrator
    
    orchestrator = RobustCodeGenerationOrchestrator()
    script = await orchestrator.generate_script("move a blue pixel from left to right")
    
    # Save the generated script
    with open("my_sketch.py", "w") as f:
        f.write(script.code)
"""

# =============================================================================
# MAIN ORCHESTRATOR CLASS
# =============================================================================

from .orchestrator import RobustCodeGenerationOrchestrator, create_robust_orchestrator

# =============================================================================
# IMPROVED EXPERT AGENTS
# =============================================================================

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
    ApplyBouncingBehaviorSchema,
    ApplyGentleMovementSchema,
    ApplyRandomMovementSchema,
    ApplyRotationalMovementSchema,
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
    
    # Enhanced script generators
    generate_bouncing_script,
    generate_circular_motion_script,
    generate_gentle_movement_script,
    generate_basic_movement_script,
    
    # Tool descriptions for agents
    PARTICLE_AGENT_TOOLS,
    COLOR_AGENT_TOOLS,
    MOTION_AGENT_TOOLS,
    PHYSICS_AGENT_TOOLS
)

# =============================================================================
# IMPROVED JSON UTILITIES
# =============================================================================

from .json_utils import (
    clean_json_response,
    fix_common_json_issues,
    safe_json_parse,
    validate_tool_call_json,
    validate_task_plan_json,
    extract_json_from_text,
    fix_quotes,
    test_json_utils,
    test_improved_json_utils,  # Fixed: now this function exists
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

__version__ = "0.3.0"  # Updated version reflecting improvements
__author__ = "Tölvera MoE System"
__description__ = "Improved Robust Mixture-of-Experts system for generating organic Tölvera sketches from natural language"

# Enhanced agent specializations
AGENT_SPECIALIZATIONS = {
    "conductor": "Enhanced master planner with better task decomposition and number extraction",
    "particle": "Improved particle creation expert with accurate count detection",
    "color": "Enhanced color management with comprehensive color detection", 
    "motion": "Improved movement expert with better direction and velocity extraction",
    "physics": "Enhanced physics expert supporting organic behaviors like flocking and bouncing",
    "composition": "Improved script assembly with physics-aware code generation"
}

# =============================================================================
# MAIN EXPORTS FOR PUBLIC API
# =============================================================================

__all__ = [
    # === PRIMARY ROBUST CLASSES ===
    "RobustCodeGenerationOrchestrator",
    "create_robust_orchestrator",
    
    # === CORE DATA STRUCTURES ===
    "GeneratedScript",
    "ToolCall",
    "TaskResult",
    "TaskPlan",
    
    # === IMPROVED EXPERT AGENTS ===
    "RobustConductorAgent",
    "RobustParticleCreationAgent",
    "RobustColorPaletteAgent",
    "RobustMotionDynamicsAgent",
    "RobustPhysicsAgent",
    "RobustCompositionAgent",
    "RobustBaseAgent",
    "create_robust_agents",
    
    # === ENHANCED TOOL SCHEMAS ===
    "CreateParticlesSchema",
    "ApplyBouncingBehaviorSchema",
    "ApplyGentleMovementSchema", 
    "ApplyRandomMovementSchema",
    "ApplyRotationalMovementSchema",
    "ApplyFlockBehaviorSchema",
    
    # === UTILITY FUNCTIONS ===
    "print_model_status",
    "print_dependency_status",
    "check_dependencies",
    "save_generated_script",
    "validate_generated_script",
    "tool_calls_to_python_code",
    
    # === IMPROVED JSON UTILITIES ===
    "clean_json_response",
    "safe_json_parse",
    "validate_tool_call_json",
    "validate_task_plan_json",
    "fix_common_json_issues",
    "extract_json_from_text",
    "test_json_utils",
    "test_improved_json_utils",  # Fixed: now included in exports
    
    # === ENHANCED SCRIPT GENERATORS ===
    "generate_bouncing_script",
    "generate_circular_motion_script", 
    "generate_gentle_movement_script",
    "generate_basic_movement_script",
    
    # === PERFORMANCE AND MONITORING ===
    "PerformanceMonitor",
    
    # === EXAMPLES AND TESTING ===
    "EXAMPLE_REQUESTS",
    "TEST_REQUESTS",
    
    # === COLOR UTILITIES ===
    "STANDARD_COLORS",
    "get_color_by_name",
    
    # === METADATA ===
    "AGENT_SPECIALIZATIONS",
    
    # === CONVENIENCE FUNCTIONS ===
    "robust_quick_generate", 
    "check_system_ready",
    "get_example_requests",
    "get_system_info",
    "create_recommended_orchestrator",
    "print_usage_recommendations",
    "print_improvement_summary"
]

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def robust_quick_generate(request: str, save_file: str = None) -> GeneratedScript:
    """
    Convenience function using the improved robust orchestrator.
    
    Args:
        request: Natural language request
        save_file: Optional filename to save the script
        
    Returns:
        GeneratedScript object
    """
    orchestrator = RobustCodeGenerationOrchestrator()
    script = await orchestrator.generate_script(request)
    
    if save_file:
        save_generated_script(script.code, save_file)
    
    return script

def check_system_ready() -> bool:
    """
    Check if the improved MoE system is ready to use.
    
    Returns:
        True if all dependencies are satisfied
    """
    return print_dependency_status()

def get_example_requests() -> list:
    """Get a list of example requests for testing the improved system."""
    enhanced_examples = [
        # Basic movement examples
        "move a blue pixel from left to right",
        "create one red particle moving upward",
        "yellow particle moving in a circle",
        
        # Multiple particle examples
        "three green particles bouncing around", 
        "five blue particles moving in different directions",
        "create several orange particles with varying speeds",
        
        # Organic behavior examples
        "particles that flock together like birds",
        "particles bouncing off the walls",
        "gentle floating particles",
        "chaotic random particle movement",
        
        # Complex examples
        "blue particles that attract each other",
        "create a swirling galaxy of particles",
        "simulate falling snow with white particles"
    ]
    return enhanced_examples

def get_system_info() -> dict:
    """Get comprehensive system information including improvements."""
    return {
        "version": __version__,
        "dependencies": check_dependencies(),
        "models": check_required_models(),
        "best_models": get_best_available_models(),
        "example_requests": len(get_example_requests()),
        "available_agents": list(AGENT_SPECIALIZATIONS.keys()),
        "robust_system": True,
        "improved_json_utilities": True,
        "enhanced_agent_routing": True,
        "organic_behaviors": True,
        "fallback_support": True,
        "javascript_code_detection": True,
        "tool_name_normalization": True,
        "physics_behaviors": ["bouncing", "flocking", "gentle_movement", "random_movement", "rotational"]
    }

# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_recommended_orchestrator() -> RobustCodeGenerationOrchestrator:
    """Create the improved robust orchestrator."""
    return RobustCodeGenerationOrchestrator()

# =============================================================================
# IMPROVED MODULE INITIALIZATION
# =============================================================================

def _check_environment():
    """Check the environment on module import with better error handling."""
    import logging
    logger = logging.getLogger(__name__)
    
    # Check basic dependencies
    try:
        deps = check_dependencies()
        
        if not deps.get("pydantic_ai", False):
            logger.warning("pydantic-ai not found. Install with: pip install pydantic-ai")
        
        if not deps.get("ollama_connection", False):
            logger.warning("Ollama not accessible. Start with: ollama serve")
        
        if not deps.get("required_models", False):
            logger.warning("Required models not found. Install with: ollama pull llama3.2:3b")
        
        # Log successful initialization
        if all(deps.values()):
            logger.info("✅ Improved MoE system fully ready with all enhancements")
    except Exception as e:
        logger.debug(f"Environment check failed: {e}")

# Run environment check on import (but don't fail if there are issues)
try:
    _check_environment()
except Exception:
    pass  # Don't fail module import due to environment checks

# =============================================================================
# USAGE RECOMMENDATIONS
# =============================================================================

def print_usage_recommendations():
    """Print usage recommendations for the improved robust system."""
    print("""
🎯 Tölvera Improved MoE System - Usage Recommendations
====================================================

RECOMMENDED (Improved Robust System):
  from tolvera.llm.compositional import RobustCodeGenerationOrchestrator
  orchestrator = RobustCodeGenerationOrchestrator()

CONVENIENCE FUNCTION (Recommended):
  from tolvera.llm.compositional import robust_quick_generate
  script = await robust_quick_generate("your request", "output.py")

KEY IMPROVEMENTS IN v0.3.0:
  ✅ Fixed JavaScript code in JSON responses
  ✅ Better agent routing (no more misrouted steps)
  ✅ Improved tool name normalization
  ✅ Enhanced organic particle behaviors
  ✅ Better number and color extraction
  ✅ Multiple fallback levels
  ✅ Physics-aware script generation
  ✅ Comprehensive error recovery

SUPPORTED ORGANIC BEHAVIORS:
  • Bouncing particles with collision detection
  • Flocking behaviors like bird murmurations  
  • Gentle floating and drifting motion
  • Chaotic random movement patterns
  • Circular and rotational motion
  • Variable speed particles

TESTING:
  Use the simplified test script to debug specific issues
  Use test_improved_json_utils() to test JSON parsing
""")

def print_improvement_summary():
    """Print a summary of all the improvements made."""
    print("""
🚀 MoE System Improvements Summary (v0.3.0)
==========================================

🔧 JSON PARSING FIXES:
  • Detects and fixes Math.random() JavaScript code
  • Normalizes tool names (create_particle → create_particles)
  • Multiple parsing strategies with aggressive cleaning
  • Better parameter sanitization

🎯 AGENT ROUTING IMPROVEMENTS:
  • Clear priority system: Composition > Physics > Motion > Color > Particle
  • Better keyword detection for each agent type
  • Context-aware fallbacks based on missing components
  • Intelligent step number fallbacks

🧠 ENHANCED AGENT PROMPTS:
  • Explicit tool name restrictions
  • JSON-only response requirements
  • Comprehensive examples for each agent
  • Better internal fallback logic

🌊 MORE ORGANIC BEHAVIORS:
  • Physics-first approach for natural motion
  • Better number and color extraction
  • Smarter velocity and direction detection
  • Context tracking for coherent generation

📋 NEW PHYSICS BEHAVIORS:
  • apply_bouncing_behavior: Collision detection and boundary bouncing
  • apply_gentle_movement: Soft floating and drifting
  • apply_random_movement: Chaotic particle motion
  • apply_rotational_movement: Circular and orbital patterns
  • apply_flock_behavior: Bird-like flocking with cohesion/separation

🛠 TESTING IMPROVEMENTS:
  • Simplified test script for focused debugging
  • Individual agent testing capabilities
  • JSON parsing issue reproduction
  • Agent routing verification tools
""")

# =============================================================================
# BACKWARDS COMPATIBILITY
# =============================================================================

# Maintain backwards compatibility with older imports
CodeGenerationOrchestrator = RobustCodeGenerationOrchestrator
quick_generate = robust_quick_generate