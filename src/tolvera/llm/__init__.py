# src/tolvera/llm/__init__.py
"""
Enhanced LLM integration for Tölvera with robust compositional MoE system.

This module provides both the original high-level sketch generation system
and the new robust compositional Mixture-of-Experts system for natural language
to code generation.
"""

# =============================================================================
# EXISTING SYSTEM IMPORTS (keep your current ones)
# =============================================================================

# Try to import existing components - adjust these based on your actual structure
try:
    from .agent import SketchAgent
except ImportError:
    SketchAgent = None

try:
    from .models import *
except ImportError:
    pass

try:
    from .code_generation import *
except ImportError:
    pass

try:
    from .utils import *
except ImportError:
    pass

# =============================================================================
# NEW: ROBUST COMPOSITIONAL MoE SYSTEM (RECOMMENDED)
# =============================================================================

# Primary robust classes (recommended for new projects)
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

# Legacy classes (for backward compatibility)
from .compositional import (
    CodeGenerationOrchestrator,
    SimpleOrchestrator,
    create_orchestrator,
    quick_generate,
)

# Core data structures (used by both systems)
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
    # === EXISTING SYSTEM (adjust based on your actual exports) ===
    "SketchAgent",  # Remove if you don't have this
    
    # === ROBUST MoE SYSTEM (RECOMMENDED) ===
    
    # Primary robust interfaces
    "RobustCodeGenerationOrchestrator",  # Main robust MoE orchestrator
    "create_robust_orchestrator",        # Factory for robust orchestrator
    "robust_quick_generate",             # Quick robust script generation
    
    # Robust expert agents
    "RobustConductorAgent",              # Robust master planner
    "RobustParticleCreationAgent",       # Robust particle creation expert
    "RobustColorPaletteAgent",           # Robust color management expert
    "RobustMotionDynamicsAgent",         # Robust movement expert
    "RobustPhysicsAgent",                # Robust physics expert
    "RobustCompositionAgent",            # Robust code assembly expert
    "create_robust_agents",              # Factory for robust agents
    
    # JSON utilities for robust parsing
    "clean_json_response",               # Clean LLM JSON responses
    "safe_json_parse",                   # Safe JSON parsing with fallbacks
    "validate_tool_call_json",           # Validate tool call structure
    "validate_task_plan_json",           # Validate task plan structure

    
    # === CORE DATA STRUCTURES ===
    "GeneratedScript",                   # Generated script container
    "ToolCall",                          # Tool call representation
    "TaskResult",                        # Agent task result
    "TaskPlan",                          # Conductor plan
    
    # === UTILITY FUNCTIONS ===
    "print_model_status",                # Check available models
    "print_dependency_status",           # Check all dependencies
    "check_dependencies",                # Dependency status dict
    "save_generated_script",             # Save script to file
    "validate_generated_script",         # Validate generated code
    "tool_calls_to_python_code",         # Direct code generation
    
    # === SYSTEM UTILITIES ===
    "check_system_ready",                # System readiness check
    "get_example_requests",              # Get example prompts
    "get_system_info",                   # System information
    
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
# MODULE METADATA
# =============================================================================

__version__ = "0.2.0"  # Updated for robust system
__description__ = "Enhanced LLM integration for Tölvera with robust compositional MoE system"

# =============================================================================
# CONVENIENCE ALIASES AND FACTORY FUNCTIONS
# =============================================================================

# Recommended entry points for new users
def create_moe_orchestrator(robust: bool = True):
    """
    Create an MoE orchestrator.
    
    Args:
        robust: Whether to use the robust version (recommended)
        
    Returns:
        Orchestrator instance
    """
    if robust:
        return RobustCodeGenerationOrchestrator()
    else:
        return CodeGenerationOrchestrator()

async def generate_script(request: str, save_file: str = None, robust: bool = True) -> GeneratedScript:
    """
    Generate a Tölvera script from natural language.
    
    Args:
        request: Natural language request
        save_file: Optional filename to save the script
        robust: Whether to use the robust system (recommended)
        
    Returns:
        GeneratedScript object
        
    Example:
        script = await generate_script("move a blue pixel from left to right", "my_sketch.py")
    """
    if robust:
        return await robust_quick_generate(request, save_file)
    else:
        return await quick_generate(request, save_file)

# Backward compatibility aliases
MoEOrchestrator = RobustCodeGenerationOrchestrator  # Recommended
SketchGenerator = RobustCodeGenerationOrchestrator  # Alternative name
LegacyMoEOrchestrator = CodeGenerationOrchestrator  # Legacy version

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
        
        # Save the script
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

3. UNIVERSAL GENERATION - Auto-selects robust system:

    from tolvera.llm import generate_script
    import asyncio
    
    async def main():
        # Uses robust system by default
        script = await generate_script(
            "blue particles bouncing around",
            "bouncing.py"
        )
        print(f"Generated: {script.title}")
    
    asyncio.run(main())

4. SYSTEM CHECK - Verify everything is ready:

    from tolvera.llm import check_system_ready, print_model_status
    
    if check_system_ready():
        print("✅ Robust system ready!")
    else:
        print("❌ Setup required")
        print_model_status()  # See what's missing

5. JSON UTILITIES - For custom agent development:

    from tolvera.llm import clean_json_response, safe_json_parse
    
    # Clean messy LLM responses
    messy_response = 'Here is JSON: {"tool": "create_particles"} Hope this helps!'
    cleaned = clean_json_response(messy_response)
    parsed = safe_json_parse(cleaned)
    
    print(f"Cleaned: {cleaned}")
    print(f"Parsed: {parsed}")

6. PERFORMANCE MONITORING:

    from tolvera.llm import PerformanceMonitor, RobustCodeGenerationOrchestrator
    import asyncio
    import time
    
    async def monitored_generation():
        monitor = PerformanceMonitor()
        orchestrator = RobustCodeGenerationOrchestrator()
        
        start_time = time.time()
        try:
            script = await orchestrator.generate_script("your request here")
            duration = time.time() - start_time
            monitor.record_request(duration, True)
            print(f"✅ Generated in {duration:.2f}s")
        except Exception as e:
            duration = time.time() - start_time
            monitor.record_request(duration, False)
            print(f"❌ Failed in {duration:.2f}s: {e}")
        
        monitor.print_summary()
    
    asyncio.run(monitored_generation())

7. LEGACY COMPATIBILITY - Using original system:

    from tolvera.llm import CodeGenerationOrchestrator  # Legacy
    import asyncio
    
    async def legacy_usage():
        orchestrator = CodeGenerationOrchestrator()  # Less robust
        script = await orchestrator.generate_script("create particles")
        print(script.code)
    
    asyncio.run(legacy_usage())

====================================
🌟 ROBUST SYSTEM ADVANTAGES:
  ✅ Better JSON parsing and error handling
  ✅ Fallback mechanisms at every level  
  ✅ Individual agent isolation
  ✅ Keyword-based fallback generation
  ✅ Direct code composition (no LLM for final step)
  ✅ Comprehensive error recovery

For more examples, see the tests/ directory!
""")

def get_quick_start_guide() -> str:
    """Get a quick start guide for new users."""
    return """
🚀 QUICK START GUIDE - Robust Tölvera MoE System
===============================================

1. Install dependencies:
   pip install pydantic-ai requests

2. Install and start Ollama:
   # Install from https://ollama.ai
   ollama serve

3. Pull required models:
   ollama pull llama3.2:3b
   ollama pull qwen2.5:3b

4. Check system status:
   from tolvera.llm import check_system_ready
   check_system_ready()

5. Generate your first script (ROBUST):
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

🎉 That's it! You're generating robust Tölvera sketches from natural language!

🔧 TROUBLESHOOTING:
   • If agents fail, the robust system automatically uses fallbacks
   • JSON parsing errors are automatically cleaned and retried
   • Ultimate fallback ensures you always get a working script
   • Use test_robust_system.py to verify functionality
"""

# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

def _initialize_module():
    """Initialize the module and check system status."""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # Quick dependency check
        deps = check_dependencies()
        
        missing_deps = [k for k, v in deps.items() if not v]
        if missing_deps:
            logger.info(f"Robust MoE system available but some dependencies missing: {missing_deps}")
            logger.info("Run tolvera.llm.print_dependency_status() for details")
        else:
            logger.info("✅ Robust MoE system fully ready")
            
    except Exception as e:
        logger.debug(f"Module initialization check failed: {e}")

# Run initialization (but don't fail if there are issues)
try:
    _initialize_module()
except Exception:
    pass  # Don't fail module import

# Add new functions to exports
__all__.extend([
    "create_moe_orchestrator",
    "generate_script",
    "print_usage_examples",
    "get_quick_start_guide",
    "MoEOrchestrator",
    "SketchGenerator", 
    "LegacyMoEOrchestrator"
])