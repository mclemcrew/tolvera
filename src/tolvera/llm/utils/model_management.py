"""
Ollama model management
"""

import logging
import requests
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

LOCALHOST = 'http://localhost:11434'


class OllamaModelManager:
    # Tool-compatible models prioritized for sketch generation (from here: https://ollama.com/search?c=tools)
    
    RECOMMENDED_MODELS = {
        # Creative = larger models that show to be more performative
        "creative": ["qwen3:8b", "llama3.1:8b", "command-r:35b", "hermes3:8b"],
        # Balance = Smaller model size but still pretty good output
        "balanced": ["qwen3:4b", "mistral:7b", "llama3.2:3b", "qwen2.5:7b"],
        # Fast = Smallest model available
        "fast": ["qwen3:1.7b", "smollm2:1.7b", "granite3-dense:2b", "qwen2.5:1.5b"],
        # Large = The largest out of the family of models.
        "large": ["command-r-plus:104b", "llama3.1:70b", "mixtral:8x22b", "qwen3:32b"]
    }

    # This list is now based on models from the user-provided source that support the OpenAI API standard for tools.
    TOOL_COMPATIBLE = [
        "deepseek-r1:8b", "qwen3:0.6b", "qwen3:1.7b", "qwen3:4b", "qwen3:8b", "qwen3:14b", "qwen3:32b",
        "devstral:24b", "llama3.3:70b", "llama3.2:1b", "llama3.2:3b", "llama3.1:8b", "llama3.1:70b", "llama3.1:405b",
        "mistral:7b", "qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:7b", "qwen2.5:14b", "qwen2.5:32b", "qwen2.5:72b",
        "qwen2:0.5b", "qwen2:1.5b", "qwen2:7b", "qwen2:72b", "mistral-nemo:12b", "mixtral:8x7b", "mixtral:8x22b",
        "smollm2:135m", "smollm2:360m", "smollm2:1.7b", "command-r:35b", "command-r-plus:104b",
        "hermes3:3b", "hermes3:8b", "hermes3:70b", "hermes3:405b", "phi4-mini:3.8b", "mistral-large:123b",
        "granite3-dense:2b", "granite3-dense:8b", "llama3-groq-tool-use:8b", "llama3-groq-tool-use:70b",
        "firefunction-v2:70b"
    ]

    AVOID_MODELS = [
        # TODO: Update as we find models that simply don't work with this sketch generator we're working on.
    ]
    
    def __init__(self, host: str = LOCALHOST):
        self.host = host
        self._available_models = None
    
    def get_available_models(self, refresh: bool = False) -> List[str]:
        if refresh or self._available_models is None:
            try:
                response = requests.get(f"{self.host}/api/tags", timeout=5)
                if response.status_code == 200:
                    self._available_models = [m['name'] for m in response.json().get('models', [])]
                else:
                    logger.warning(f"Ollama returned status {response.status_code}")
                    self._available_models = []
            except requests.RequestException as e:
                logger.warning(f"Could not connect to Ollama: {e}")
                self._available_models = []
        
        return self._available_models or []
    
    def get_recommended_models(self, category: str = "balanced") -> List[str]:
        available = self.get_available_models()
        recommended = self.RECOMMENDED_MODELS.get(category, self.RECOMMENDED_MODELS["balanced"])
        
        # Return available models from recommended list
        return [model for model in recommended if model in available]
    
    def is_model_compatible(self, model_name: str) -> bool:
        if model_name in self.AVOID_MODELS:
            return False
        
        if model_name in self.TOOL_COMPATIBLE:
            return True
        
        # For unknown models, assume compatible
        available = self.get_available_models()
        return model_name in available
    
    def select_best_model(self, preferred: Optional[str] = None) -> Optional[str]:
        available = self.get_available_models()
        
        if not available:
            logger.warning("No Ollama models available")
            return None
        
        # If preferred model is available and compatible, use it
        if preferred and preferred in available and self.is_model_compatible(preferred):
            logger.info(f"Using preferred model: {preferred}")
            return preferred
        
        # Try recommended models in order of preference
        for category in ["creative", "balanced", "fast"]:
            recommended = self.get_recommended_models(category)
            if recommended:
                best = recommended[0]
                logger.info(f"Using recommended {category} model: {best}")
                return best
        
        # Fallback to any compatible model
        for model in available:
            if self.is_model_compatible(model):
                logger.info(f"Using fallback compatible model: {model}")
                return model
        
        # Last resort - use any available model they have
        if available:
            fallback = available[0]
            logger.warning(f"Using unvalidated fallback model: {fallback}")
            return fallback
        
        return None
    
    # This is a pydantic thing working with ollama
    def test_openai_endpoint(self) -> bool:
        try:
            response = requests.get(f"{self.host}/v1/models", timeout=5)
            if response.status_code == 200:
                logger.debug("✅ Ollama OpenAI endpoint working")
                return True
            else:
                logger.warning(f"⚠️  OpenAI endpoint returned {response.status_code}")
                return False
        except requests.RequestException as e:
            logger.warning(f"⚠️  OpenAI endpoint test failed: {e}")
            return False
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                for model in models:
                    if model['name'] == model_name:
                        return model
        except requests.RequestException:
            pass
        return None
    
    def validate_setup(self) -> Dict[str, Any]:
        validation = {
            "ollama_running": False,
            "openai_endpoint": False,
            "models_available": 0,
            "compatible_models": 0,
            "recommended_model": None,
            "suggestions": []
        }
        
        # Test connection
        available = self.get_available_models(refresh=True)
        validation["ollama_running"] = len(available) > 0
        validation["models_available"] = len(available)
        
        if not validation["ollama_running"]:
            validation["suggestions"].append("Start Ollama service: 'ollama serve'")
            return validation
        
        # Test endpoint
        validation["openai_endpoint"] = self.test_openai_endpoint()
        
        # Count compatible models
        compatible = [m for m in available if self.is_model_compatible(m)]
        validation["compatible_models"] = len(compatible)
        
        # Get best model
        validation["recommended_model"] = self.select_best_model()
        
        # Generate suggestions
        if validation["compatible_models"] == 0:
            validation["suggestions"].extend([
                "Install a compatible model:",
                "  ollama pull <model_name>",
            ])
        elif validation["compatible_models"] < 2:
            # If they only have one model installed, might want to suggest another one just in case it fails occasionally
            validation["suggestions"].append("Consider installing additional models for better options")
        
        return validation
    
    def ensure_compatible_model(self, preferred: Optional[str] = None) -> str:
        """
        Vibe check to know that they have a compatible model that's downloaded and ready to go.
        """
        available = self.get_available_models(refresh=True)
        
        # Check if preferred model is available and compatible
        if preferred and preferred in available and self.is_model_compatible(preferred):
            logger.info(f"✅ Using preferred model: {preferred}")
            return preferred
        
        # Find any compatible model that's already available
        for model in self.TOOL_COMPATIBLE:
            if model in available:
                logger.info(f"✅ Using available compatible model: {model}")
                return model
        
        # No compatible models available - try to download one
        logger.warning("No compatible models found - attempting to download...")
        
        # Try to download "best" models
        download_candidates = [
            "qwen3:4b",        
            "qwen2.5:7b",      
            "llama3.1:8b",     
            "mistral:7b",      
            "qwen3:1.7b"       
        ]
        
        for model in download_candidates:
            if self.download_model(model):
                logger.info(f"✅ Successfully downloaded and using: {model}")
                return model
        
        # If all downloads fail, return any available model as last resort
        if available:
            fallback = available[0]
            logger.warning(f"⚠️  Using fallback model: {fallback}")
            return fallback
        
        raise RuntimeError("No models available and download failed")
    
    def download_model(self, model_name: str) -> bool:
        """
        I'm using this in case the user hasn't downloaded any model but wants to use this tool still.  We'll just download it for them.
        """
        try:
            import subprocess
            logger.info(f"Downloading model: {model_name}...")
            
            result = subprocess.run(
                ["ollama", "pull", model_name],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                logger.info(f"✅ Successfully downloaded: {model_name}")
                self._available_models = None
                return True
            else:
                logger.error(f"❌ Download failed for {model_name}: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Download timeout for {model_name}")
            return False
        except Exception as e:
            logger.error(f"❌ Download error for {model_name}: {e}")
            return False


# TODO clean this up.  In the test_agent I was using these to make sure everything was working rather than instantiating the class.
def get_available_models(host: str = LOCALHOST) -> List[str]:
    """Get list of available Ollama models."""
    manager = OllamaModelManager(host)
    return manager.get_available_models()


def get_best_available_model(host: str = LOCALHOST) -> Optional[str]:
    manager = OllamaModelManager(host)
    return manager.select_best_model()


def check_ollama_connection(host: str = LOCALHOST) -> bool:
    try:
        response = requests.get(f"{host}/api/tags", timeout=3)
        return response.status_code == 200
    except requests.RequestException:
        return False