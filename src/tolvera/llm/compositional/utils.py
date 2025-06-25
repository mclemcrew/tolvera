"""
These were some utilies that we had used previously, but most of these are not used anymore.
"""

import logging
import requests
from typing import Dict, List
from pathlib import Path

logger = logging.getLogger(__name__)

def check_ollama_connection() -> bool:
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Ollama connection failed: {e}")
        return False

def get_available_models() -> List[str]:
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
    required_models = [
        "qwen2.5:3b:latest",  # For Conductor and Composition
        "qwen2.5:3b",   # For expert agents
    ]
    
    # Also check for alternative models
    alternative_models = [
        "llama3.1:8b",  # Better alternative for Conductor
        "qwen2.5:3b",   # Better alternative for experts
    ]
    
    available_models = get_available_models()
    model_status = {}
    
    for model in required_models + alternative_models:
        model_status[model] = model in available_models
    
    return model_status

def get_best_available_models() -> Dict[str, str]:
    """Get the best available models for each agent type."""
    available = get_available_models()
    
    conductor_models = ["llama3.1:8b", "qwen2.5:3b:latest", "qwen2.5:3b", "qwen2.5:3b"]
    expert_models = ["qwen2.5:3b", "qwen2.5:3b", "qwen2.5:3b:latest"]
    
    best_models = {}
    
    for model in conductor_models:
        if model in available:
            best_models["conductor"] = model
            best_models["composition"] = model
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
    print("Checking Ollama Model Status...")
    print("=" * 50)
    
    if not check_ollama_connection():
        print("❌ Ollama server is not running or not accessible")
        print("   Please start Ollama with: ollama serve")
        return False
    
    print("✅ Ollama server is running")
    
    model_status = check_required_models()
    available_models = get_available_models()
    
    print(f"\nAvailable Models ({len(available_models)} total):")
    for model in available_models:
        print(f"   ✅ {model}")
    
    print(f"\n Required Models Status:")
    required = ["qwen2.5:3b:latest", "qwen2.5:3b"]
    for model in required:
        status = "✅" if model_status.get(model, False) else "❌"
        print(f"   {status} {model}")
        if not model_status.get(model, False):
            print(f"      Install with: ollama pull {model}")
    
    print(f"\n Recommended Models Status:")
    recommended = ["llama3.1:8b", "qwen2.5:3b"]
    for model in recommended:
        status = "✅" if model_status.get(model, False) else "⚠️"
        print(f"   {status} {model}")
        if not model_status.get(model, False):
            print(f"      Install with: ollama pull {model}")
    
    best_models = get_best_available_models()
    print(f"\nBest Available Models for MoE System:")
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

def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_generated_script(script_content: str, filename: str, output_dir: str = "generated_sketches") -> Path:

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
    test_dir = Path("tests/generated_sketches")
    ensure_directory(test_dir)
    return test_dir
