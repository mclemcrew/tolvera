# src/tolvera/llm/compositional/utils.py
"""
Utility functions for the MoE system.
Includes model checking, validation, and helper functions.
"""

import logging
import requests
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# =============================================================================
# MODEL AND DEPENDENCY CHECKING
# =============================================================================

def check_ollama_connection() -> bool:
    """Check if Ollama server is running and accessible."""
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Ollama connection failed: {e}")
        return False

def get_available_models() -> List[str]:
    """Get list of available models from Ollama."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            models_data = response.json()
            return [model["name"] for model in models_data.get("models", [])]
        else:
            logger.error(f"Failed to get models: HTTP {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"Failed to get available models: {e}")
        return []

def check_required_models() -> Dict[str, bool]:
    """Check if required models are available."""
    required_models = [
        "llama3.2:3b",  # For Conductor and Composition
        "qwen2.5:3b",   # For expert agents
    ]
    
    # Also check for alternative models
    alternative_models = [
        "llama3.1:8b",  # Better alternative for Conductor
        "qwen2.5:7b",   # Better alternative for experts
    ]
    
    available_models = get_available_models()
    model_status = {}
    
    for model in required_models + alternative_models:
        model_status[model] = model in available_models
    
    return model_status

def get_best_available_models() -> Dict[str, str]:
    """Get the best available models for each agent type."""
    available = get_available_models()
    
    # Preferred models for each role
    conductor_models = ["llama3.1:8b", "llama3.2:3b", "qwen2.5:7b", "qwen2.5:3b"]
    expert_models = ["qwen2.5:7b", "qwen2.5:3b", "llama3.2:3b"]
    
    best_models = {}
    
    # Find best conductor model
    for model in conductor_models:
        if model in available:
            best_models["conductor"] = model
            best_models["composition"] = model  # Same model for composition
            break
    else:
        best_models["conductor"] = "qwen2.5:3b"  # Fallback
        best_models["composition"] = "qwen2.5:3b"
    
    # Find best expert model
    for model in expert_models:
        if model in available:
            best_models["expert"] = model
            break
    else:
        best_models["expert"] = "qwen2.5:3b"  # Fallback
    
    return best_models

def print_model_status():
    """Print a detailed status report of available models."""
    print("🔍 Checking Ollama Model Status...")
    print("=" * 50)
    
    if not check_ollama_connection():
        print("❌ Ollama server is not running or not accessible")
        print("   Please start Ollama with: ollama serve")
        return False
    
    print("✅ Ollama server is running")
    
    model_status = check_required_models()
    available_models = get_available_models()
    
    print(f"\n📦 Available Models ({len(available_models)} total):")
    for model in available_models:
        print(f"   ✅ {model}")
    
    print(f"\n🎯 Required Models Status:")
    required = ["llama3.2:3b", "qwen2.5:3b"]
    for model in required:
        status = "✅" if model_status.get(model, False) else "❌"
        print(f"   {status} {model}")
        if not model_status.get(model, False):
            print(f"      Install with: ollama pull {model}")
    
    print(f"\n🚀 Recommended Models Status:")
    recommended = ["llama3.1:8b", "qwen2.5:7b"]
    for model in recommended:
        status = "✅" if model_status.get(model, False) else "⚠️"
        print(f"   {status} {model}")
        if not model_status.get(model, False):
            print(f"      Install with: ollama pull {model}")
    
    best_models = get_best_available_models()
    print(f"\n🎭 Best Available Models for MoE System:")
    print(f"   Conductor: {best_models['conductor']}")
    print(f"   Experts: {best_models['expert']}")
    print(f"   Composition: {best_models['composition']}")
    
    # Check if we have minimum requirements
    has_minimum = any(model_status.get(model, False) for model in required)
    
    if has_minimum:
        print(f"\n✅ System ready to run!")
    else:
        print(f"\n❌ Missing required models. Install at least one of:")
        for model in required:
            print(f"   ollama pull {model}")
    
    return has_minimum

# =============================================================================
# FILE AND PATH UTILITIES
# =============================================================================

def ensure_directory(path: Path) -> Path:
    """Ensure directory exists, create if it doesn't."""
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_generated_script(script_content: str, filename: str, output_dir: str = "generated_sketches") -> Path:
    """Save a generated script to a file."""
    output_path = Path(output_dir)
    ensure_directory(output_path)
    
    # Ensure .py extension
    if not filename.endswith(".py"):
        filename += ".py"
    
    file_path = output_path / filename
    
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(script_content)
        logger.info(f"✅ Saved script to: {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"❌ Failed to save script to {file_path}: {e}")
        raise

def create_test_directory() -> Path:
    """Create and return the test output directory."""
    test_dir = Path("tests/generated_sketches")
    ensure_directory(test_dir)
    return test_dir

def sanitize_filename(name: str) -> str:
    """Sanitize a string to be a valid filename."""
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, '_')
    
    # Replace spaces with underscores
    name = name.replace(' ', '_')
    
    # Remove multiple consecutive underscores
    while '__' in name:
        name = name.replace('__', '_')
    
    # Trim underscores from start and end
    name = name.strip('_')
    
    # Ensure it's not empty
    if not name:
        name = "generated_script"
    
    # Limit length
    if len(name) > 50:
        name = name[:50]
    
    return name

# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

def validate_script_syntax(script_content: str) -> Tuple[bool, Optional[str]]:
    """Validate that the generated script has valid Python syntax."""
    try:
        compile(script_content, '<generated_script>', 'exec')
        return True, None
    except SyntaxError as e:
        error_msg = f"Syntax error at line {e.lineno}: {e.msg}"
        return False, error_msg
    except Exception as e:
        return False, f"Compilation error: {str(e)}"

def check_tolvera_imports(script_content: str) -> bool:
    """Check if the script contains proper Tölvera imports."""
    required_imports = ["from tolvera import", "import taichi"]
    return all(imp in script_content for imp in required_imports)

def validate_generated_script(script_content: str) -> Dict[str, Any]:
    """Comprehensive validation of a generated script."""
    validation_results = {
        "valid_syntax": False,
        "has_tolvera_imports": False,
        "has_main_function": False,
        "has_render_function": False,
        "has_taichi_kernels": False,
        "syntax_error": None,
        "warnings": [],
        "score": 0
    }
    
    # Check syntax
    syntax_valid, syntax_error = validate_script_syntax(script_content)
    validation_results["valid_syntax"] = syntax_valid
    validation_results["syntax_error"] = syntax_error
    
    # Check imports
    validation_results["has_tolvera_imports"] = check_tolvera_imports(script_content)
    if not validation_results["has_tolvera_imports"]:
        validation_results["warnings"].append("Missing required Tölvera imports")
    
    # Check for main function
    validation_results["has_main_function"] = "def main(" in script_content
    if not validation_results["has_main_function"]:
        validation_results["warnings"].append("Missing main() function")
    
    # Check for render function
    validation_results["has_render_function"] = "@tv.render" in script_content
    if not validation_results["has_render_function"]:
        validation_results["warnings"].append("Missing @tv.render decorator")
    
    # Check for Taichi kernels
    validation_results["has_taichi_kernels"] = "@ti.kernel" in script_content
    if not validation_results["has_taichi_kernels"]:
        validation_results["warnings"].append("No Taichi kernels found (may be okay for simple scripts)")
    
    # Calculate overall score
    score = 0
    if validation_results["valid_syntax"]:
        score += 40
    if validation_results["has_tolvera_imports"]:
        score += 20
    if validation_results["has_main_function"]:
        score += 20
    if validation_results["has_render_function"]:
        score += 15
    if validation_results["has_taichi_kernels"]:
        score += 5
    
    validation_results["score"] = score
    
    return validation_results

# =============================================================================
# LOGGING AND DEBUG UTILITIES
# =============================================================================

def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration for the MoE system."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    # Reduce pydantic_ai verbosity
    logging.getLogger("pydantic_ai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

def log_agent_performance(agent_name: str, task: str, duration: float, success: bool):
    """Log performance metrics for agents."""
    status = "SUCCESS" if success else "FAILED"
    logger.info(f"🎭 {agent_name}: {task} - {status} in {duration:.2f}s")

# =============================================================================
# EXAMPLE REQUESTS AND TEMPLATES
# =============================================================================

EXAMPLE_REQUESTS = [
    "move a blue pixel from left to right",
    "create three red particles that move upward",
    "make particles that flock together",
    "create a swirling galaxy effect",
    "simulate falling snow",
    "create particles that bounce off the walls",
    "make a rainbow of particles moving in different directions",
    "create particles that chase the mouse cursor",
    "simulate a simple ecosystem with predators and prey",
    "create a particle fountain effect"
]

TEST_REQUESTS = [
    # Basic movement
    "move a blue pixel from left to right",
    "create a red particle that moves upward",
    
    # Multiple particles
    "create three green particles",
    "make five yellow particles move in different directions",
    
    # Varying speeds
    "create particles with varying speeds",
    "make three blue particles move from left to right with different speeds",
    
    # Colors
    "create rainbow colored particles",
    "make particles change color over time",
    
    # Physics
    "create particles that flock together",
    "make particles bounce off the screen edges",
    "simulate gravity pulling particles down"
]

# =============================================================================
# PERFORMANCE MONITORING
# =============================================================================

class PerformanceMonitor:
    """Monitor performance of the MoE system."""
    
    def __init__(self):
        self.metrics = {
            "requests_processed": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "total_time": 0.0,
            "average_time": 0.0,
            "agent_calls": {},
        }
    
    def record_request(self, duration: float, success: bool):
        """Record a request completion."""
        self.metrics["requests_processed"] += 1
        self.metrics["total_time"] += duration
        
        if success:
            self.metrics["successful_generations"] += 1
        else:
            self.metrics["failed_generations"] += 1
        
        # Update average
        self.metrics["average_time"] = (
            self.metrics["total_time"] / self.metrics["requests_processed"]
        )
    
    def record_agent_call(self, agent_name: str, duration: float):
        """Record an agent call."""
        if agent_name not in self.metrics["agent_calls"]:
            self.metrics["agent_calls"][agent_name] = {
                "count": 0,
                "total_time": 0.0,
                "average_time": 0.0
            }
        
        agent_metrics = self.metrics["agent_calls"][agent_name]
        agent_metrics["count"] += 1
        agent_metrics["total_time"] += duration
        agent_metrics["average_time"] = agent_metrics["total_time"] / agent_metrics["count"]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        success_rate = 0.0
        if self.metrics["requests_processed"] > 0:
            success_rate = (
                self.metrics["successful_generations"] / 
                self.metrics["requests_processed"] * 100
            )
        
        return {
            **self.metrics,
            "success_rate": success_rate
        }
    
    def print_summary(self):
        """Print performance summary."""
        summary = self.get_summary()
        
        print("\n📊 MoE System Performance Summary")
        print("=" * 40)
        print(f"Requests Processed: {summary['requests_processed']}")
        print(f"Successful: {summary['successful_generations']}")
        print(f"Failed: {summary['failed_generations']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Average Time: {summary['average_time']:.2f}s")
        
        if summary["agent_calls"]:
            print(f"\n🎭 Agent Performance:")
            for agent, metrics in summary["agent_calls"].items():
                print(f"  {agent}: {metrics['count']} calls, {metrics['average_time']:.2f}s avg")

# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

def create_integration_example() -> str:
    """Create an example of how to integrate the MoE system."""
    return '''
# Example: Integrating MoE System with Existing Tölvera Code

from tolvera.llm.compositional import CodeGenerationOrchestrator
import asyncio

async def generate_sketch_example():
    """Example of generating a sketch from natural language."""
    
    # Initialize the orchestrator
    orchestrator = CodeGenerationOrchestrator()
    
    # Generate a script from natural language
    user_request = "move a blue pixel from left to right"
    script = await orchestrator.generate_script(user_request)
    
    # Save the generated script
    with open("my_generated_sketch.py", "w") as f:
        f.write(script.code)
    
    print(f"Generated: {script.title}")
    print(f"Saved to: my_generated_sketch.py")
    
    # The generated script can now be run independently:
    # python my_generated_sketch.py

# Run the example
if __name__ == "__main__":
    asyncio.run(generate_sketch_example())
'''

def check_dependencies() -> Dict[str, bool]:
    """Check all required dependencies for the MoE system."""
    dependencies = {}
    
    # Check pydantic-ai
    try:
        import pydantic_ai
        dependencies["pydantic_ai"] = True
    except ImportError:
        dependencies["pydantic_ai"] = False
    
    # Check requests
    try:
        import requests
        dependencies["requests"] = True
    except ImportError:
        dependencies["requests"] = False
    
    # Check Ollama connection
    dependencies["ollama_connection"] = check_ollama_connection()
    
    # Check required models
    model_status = check_required_models()
    dependencies["required_models"] = any(
        model_status.get(model, False) 
        for model in ["llama3.2:3b", "qwen2.5:3b"]
    )
    
    return dependencies

def print_dependency_status():
    """Print comprehensive dependency status."""
    print("🔍 Checking MoE System Dependencies...")
    print("=" * 50)
    
    deps = check_dependencies()
    
    for dep, status in deps.items():
        icon = "✅" if status else "❌"
        print(f"{icon} {dep}")
        
        if not status:
            if dep == "pydantic_ai":
                print("   Install with: pip install pydantic-ai")
            elif dep == "requests":
                print("   Install with: pip install requests")
            elif dep == "ollama_connection":
                print("   Start Ollama with: ollama serve")
            elif dep == "required_models":
                print("   Install models with: ollama pull llama3.2:3b")
    
    all_ready = all(deps.values())
    
    if all_ready:
        print("\n✅ All dependencies satisfied! MoE system ready to use.")
    else:
        print("\n❌ Some dependencies missing. Please install missing components.")
    
    return all_ready