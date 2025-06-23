# src/tolvera/llm/__init__.py
"""
Enhanced LLM integration for Tölvera with robust compositional MoE system.

This module provides the new robust compositional Mixture-of-Experts system 
for natural language to code generation.
"""

# =============================================================================
# ROBUST COMPOSITIONAL MoE SYSTEM (RECOMMENDED)
# =============================================================================

# Primary robust classes
from .compositional import (
    RobustCodeGenerationOrchestrator,
    create_robust_orchestrator,
    robust_quick_generate,
    
    # Robust expert agents
    RobustConductorAgent,
    RobustParticleCreationAgent,
    RobustColorPaletteAgent, 
    RobustMotionDynamicsAgent,
    RobustPhysicsAgent,
    RobustCompositionAgent,
    create_robust_agents,
    
    # JSON utilities for robust parsing
    clean_json_response,
    safe_json_parse,
    validate_tool_call_json,
    validate_task_plan_json,
)

# Core data structures
from .compositional import (
    GeneratedScript,
    ToolCall,
    TaskResult,
    TaskPlan,
    
    # Tool definitions
    STANDARD_COLORS,
    get_color_by_name,
    tool_calls_to_python_code,
)

# Utility functions
from .compositional import (
    print_model_status,
    print_dependency_status,
    check_dependencies,
    save_generated_script,
    validate_generated_script,
    
    # System info and examples
    check_system_ready,
    get_example_requests,
    get_system_info,
    
    # Performance monitoring
    PerformanceMonitor,
    
    # Example data
    EXAMPLE_REQUESTS,
    TEST_REQUESTS,
    
    # Agent info
    AGENT_SPECIALIZATIONS,
)

# =============================================================================
# PUBLIC API EXPORTS
# =============================================================================

__all__ = [
    # === ROBUST MoE SYSTEM ===
    
    # Primary robust interfaces
    "RobustCodeGenerationOrchestrator",
    "create_robust_orchestrator",
    "robust_quick_generate",
    
    # Robust expert agents
    "RobustConductorAgent",
    "RobustParticleCreationAgent",
    "RobustColorPaletteAgent",
    "RobustMotionDynamicsAgent",
    "RobustPhysicsAgent",
    "RobustCompositionAgent",
    "create_robust_agents",
    
    # JSON utilities for robust parsing
    "clean_json_response",
    "safe_json_parse",
    "validate_tool_call_json",
    "validate_task_plan_json",

    
    # === CORE DATA STRUCTURES ===
    "GeneratedScript",
    "ToolCall",
    "TaskResult",
    "TaskPlan",
    
    # === UTILITY FUNCTIONS ===
    "print_model_status",
    "print_dependency_status",
    "check_dependencies",
    "save_generated_script",
    "validate_generated_script",
    "tool_calls_to_python_code",
    
    # === SYSTEM UTILITIES ===
    "check_system_ready",
    "get_example_requests",
    "get_system_info",
    
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
]

# =============================================================================
# MODULE METADATA
# =============================================================================

__version__ = "0.2.0"
__description__ = "Enhanced LLM integration for Tölvera with robust compositional MoE system"

# =============================================================================
# CONVENIENCE ALIASES AND FACTORY FUNCTIONS
# =============================================================================

def create_moe_orchestrator():
    """
    Create a robust MoE orchestrator.
    """
    return RobustCodeGenerationOrchestrator()

async def generate_script(request: str, save_file: str = None) -> GeneratedScript:
    """
    Generate a Tölvera script from natural language using the robust system.
    
    Args:
        request: Natural language request
        save_file: Optional filename to save the script
        
    Returns:
        GeneratedScript object
        
    Example:
        script = await generate_script("move a blue pixel from left to right", "my_sketch.py")
    """
    return await robust_quick_generate(request, save_file)

# =============================================================================
# USAGE EXAMPLES AND DOCUMENTATION
# =============================================================================

def print_usage_examples():
    """Print usage examples for the robust MoE system."""
    print("""
🎭 Tölvera Robust MoE System Usage Examples
===========================================

1. RECOMMENDED USAGE - Robust system with automatic fallbacks:

    from tolvera.llm import RobustCodeGenerationOrchestrator
    import asyncio
    
    async def main():
        orchestrator = RobustCodeGenerationOrchestrator()
        script = await orchestrator.generate_script("move a blue pixel from left to right")
        
        with open("my_sketch.py", "w") as f:
            f.write(script.code)
        
        print(f"Generated: {script.title}")
    
    asyncio.run(main())

2. QUICK GENERATION - Using robust convenience function:

    from tolvera.llm import robust_quick_generate
    import asyncio
    
    async def main():
        script = await robust_quick_generate(
            "create three red particles that move upward",
            save_file="red_particles.py"
        )
        print(f"Generated and saved: {script.title}")
    
    asyncio.run(main())

3. SYSTEM CHECK - Verify everything is ready:

    from tolvera.llm import check_system_ready, print_model_status
    
    if check_system_ready():
        print("✅ Robust system ready!")
    else:
        print("❌ Setup required")
        print_model_status()
""")

def get_quick_start_guide() -> str:
    """Get a quick start guide for new users."""
    return """
🚀 QUICK START GUIDE - Robust Tölvera MoE System
===============================================

1. Install dependencies:
   pip install pydantic-ai requests

2. Install and start Ollama:
   ollama serve

3. Pull required models:
   ollama pull llama3.2:3b
   ollama pull qwen2.5:3b

4. Check system status:
   from tolvera.llm import check_system_ready
   check_system_ready()

5. Generate your first script:
   from tolvera.llm import robust_quick_generate
   import asyncio
   
   async def main():
       script = await robust_quick_generate(
           "move a blue pixel from left to right",
           "my_first_sketch.py"
       )
       print("Generated:", script.title)
   
   asyncio.run(main())

6. Run your generated script:
   python my_first_sketch.py
"""

# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

def _initialize_module():
    """Initialize the module and check system status."""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        deps = check_dependencies()
        
        missing_deps = [k for k, v in deps.items() if not v]
        if missing_deps:
            logger.info(f"Robust MoE system available but some dependencies missing: {missing_deps}")
            logger.info("Run tolvera.llm.print_dependency_status() for details")
        else:
            logger.info("✅ Robust MoE system fully ready")
            
    except Exception as e:
        logger.debug(f"Module initialization check failed: {e}")

try:
    _initialize_module()
except Exception:
    pass

__all__.extend([
    "create_moe_orchestrator",
    "generate_script",
    "print_usage_examples",
    "get_quick_start_guide",
])