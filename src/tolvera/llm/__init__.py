# src/tolvera/llm/__init__.py
"""
Enhanced LLM integration for Tölvera with compositional MoE system.

This module provides both the original high-level sketch generation system
and the new compositional Mixture-of-Experts system for natural language
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
# NEW: COMPOSITIONAL MoE SYSTEM
# =============================================================================

# Main classes users will interact with
from .compositional import (
    CodeGenerationOrchestrator,
    SimpleOrchestrator,
    create_orchestrator,
    
    # Core data structures
    GeneratedScript,
    ToolCall,
    TaskResult,
    TaskPlan,
    
    # Utility functions
    print_model_status,
    print_dependency_status,
    check_dependencies,
    save_generated_script,
    validate_generated_script,
    
    # Convenience functions
    quick_generate,
    check_system_ready,
    get_example_requests,
    get_system_info,
    
    # Performance monitoring
    PerformanceMonitor,
    
    # Example data
    EXAMPLE_REQUESTS,
    TEST_REQUESTS,
    
    # Color utilities
    STANDARD_COLORS,
    get_color_by_name,
)

# Expert agents for advanced usage
from .compositional import (
    ConductorAgent,
    ParticleCreationAgent,
    ColorPaletteAgent,
    MotionDynamicsAgent,
    PhysicsAgent,
    CompositionAgent,
    AGENT_SPECIALIZATIONS,
)

# =============================================================================
# PUBLIC API EXPORTS
# =============================================================================

__all__ = [
    # === EXISTING SYSTEM (adjust based on your actual exports) ===
    "SketchAgent",  # Remove if you don't have this
    
    # === NEW COMPOSITIONAL MoE SYSTEM ===
    
    # Primary user interfaces
    "CodeGenerationOrchestrator",  # Main MoE orchestrator
    "SimpleOrchestrator",          # Simplified version for testing
    "create_orchestrator",         # Factory function
    
    # Convenience functions
    "quick_generate",              # Quick script generation
    "check_system_ready",          # System readiness check
    "get_example_requests",        # Get example prompts
    "get_system_info",             # System information
    
    # Core data structures
    "GeneratedScript",             # Generated script container
    "ToolCall",                    # Tool call representation
    "TaskResult",                  # Agent task result
    "TaskPlan",                    # Conductor plan
    
    # System utilities
    "print_model_status",          # Check available models
    "print_dependency_status",     # Check all dependencies
    "check_dependencies",          # Dependency status dict
    "save_generated_script",       # Save script to file
    "validate_generated_script",   # Validate generated code
    
    # Performance and monitoring
    "PerformanceMonitor",          # Performance tracking
    
    # Examples and testing
    "EXAMPLE_REQUESTS",            # Example prompts for testing
    "TEST_REQUESTS",               # Test prompts
    
    # Color utilities
    "STANDARD_COLORS",             # Standard color definitions
    "get_color_by_name",           # Get color by name
    
    # Expert agents (for advanced usage)
    "ConductorAgent",              # Master planner
    "ParticleCreationAgent",       # Particle creation expert
    "ColorPaletteAgent",           # Color management expert
    "MotionDynamicsAgent",         # Movement expert
    "PhysicsAgent",                # Physics expert
    "CompositionAgent",            # Code assembly expert
    "AGENT_SPECIALIZATIONS",       # Agent role descriptions
]

# =============================================================================
# MODULE METADATA
# =============================================================================

__version__ = "0.1.0"
__description__ = "Enhanced LLM integration for Tölvera with compositional MoE system"

# =============================================================================
# USAGE EXAMPLES AND DOCUMENTATION
# =============================================================================

def print_usage_examples():
    """Print usage examples for the MoE system."""
    print("""
🎭 Tölvera MoE System Usage Examples
====================================

1. BASIC USAGE - Generate a script from natural language:

    from tolvera.llm import CodeGenerationOrchestrator
    import asyncio
    
    async def main():
        orchestrator = CodeGenerationOrchestrator()
        script = await orchestrator.generate_script("move a blue pixel from left to right")
        
        # Save the script
        with open("my_sketch.py", "w") as f:
            f.write(script.code)
        
        print(f"Generated: {script.title}")
    
    asyncio.run(main())

2. QUICK GENERATION - Using convenience function:

    from tolvera.llm import quick_generate
    import asyncio
    
    async def main():
        script = await quick_generate(
            "create three red particles that move upward",
            save_file="red_particles.py"
        )
        print(f"Generated and saved: {script.title}")
    
    asyncio.run(main())

3. SYSTEM CHECK - Verify everything is ready:

    from tolvera.llm import check_system_ready, print_model_status
    
    if check_system_ready():
        print("✅ System ready!")
    else:
        print("❌ Setup required")
        print_model_status()  # See what's missing

4. BATCH GENERATION - Generate multiple scripts:

    from tolvera.llm import CodeGenerationOrchestrator, get_example_requests
    import asyncio
    
    async def batch_generate():
        orchestrator = CodeGenerationOrchestrator()
        requests = get_example_requests()[:3]  # First 3 examples
        
        for i, request in enumerate(requests):
            script = await orchestrator.generate_script(request)
            filename = f"generated_script_{i+1}.py"
            
            with open(filename, "w") as f:
                f.write(script.code)
            
            print(f"Generated: {filename}")
    
    asyncio.run(batch_generate())

5. VALIDATION - Check generated scripts:

    from tolvera.llm import validate_generated_script
    
    with open("my_script.py", "r") as f:
        script_content = f.read()
    
    validation = validate_generated_script(script_content)
    print(f"Script validation score: {validation['score']}/100")
    
    if validation['warnings']:
        print("Warnings:", validation['warnings'])

6. PERFORMANCE MONITORING:

    from tolvera.llm import PerformanceMonitor, CodeGenerationOrchestrator
    import asyncio
    import time
    
    async def monitored_generation():
        monitor = PerformanceMonitor()
        orchestrator = CodeGenerationOrchestrator()
        
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

7. ADVANCED - Using specific agents:

    from tolvera.llm import ConductorAgent, CompositionAgent
    import asyncio
    
    async def advanced_usage():
        conductor = ConductorAgent()
        composition = CompositionAgent()
        
        # Get a plan
        plan = await conductor.plan_task("create a swirling galaxy")
        print(f"Plan: {plan.description}")
        
        # Create some basic tool calls (normally done by expert agents)
        tool_calls = [...]  # Your tool calls here
        
        # Generate final script
        script = await composition.compose_script("swirling galaxy", tool_calls)
        print(script.code)
    
    asyncio.run(advanced_usage())

====================================
For more examples, see the tests/ directory!
""")

def get_quick_start_guide() -> str:
    """Get a quick start guide for new users."""
    return """
🚀 QUICK START GUIDE
====================

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

5. Generate your first script:
   from tolvera.llm import quick_generate
   import asyncio
   
   async def main():
       script = await quick_generate(
           "move a blue pixel from left to right",
           "my_first_sketch.py"
       )
       print("Generated:", script.title)
   
   asyncio.run(main())

6. Run your generated script:
   python my_first_sketch.py

That's it! You're generating Tölvera sketches from natural language! 🎉
"""

# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================

# Provide aliases for backward compatibility if needed
MoEOrchestrator = CodeGenerationOrchestrator  # Alternative name
SketchGenerator = CodeGenerationOrchestrator  # Alternative name

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
            logger.info(f"MoE system available but some dependencies missing: {missing_deps}")
            logger.info("Run tolvera.llm.print_dependency_status() for details")
        else:
            logger.info("✅ MoE system fully ready")
            
    except Exception as e:
        logger.debug(f"Module initialization check failed: {e}")

# Run initialization (but don't fail if there are issues)
try:
    _initialize_module()
except Exception:
    pass  # Don't fail module import