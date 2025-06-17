"""
Simplified prompt building functions - Focus on organic discovery.

Minimal guidance that lets the LLM explore naturally rather than prescribing solutions.
"""

from typing import Optional
from ..models import MultiBehaviorSketchConfig


def build_generation_prompt(description: str, 
                          style: str = "organic",
                          performance_target: str = "balanced") -> str:
    """Build a generation prompt that encourages organic discovery."""
    
    prompt = f"Create a beautiful Tölvera sketch: {description}"
    
    # Minimal style guidance - just set the mood, let LLM discover
    if style == "organic":
        prompt += "\nFocus on natural, life-like movement and interactions."
    elif style == "energetic":
        prompt += "\nCreate dynamic, vibrant, fast-moving interactions."
    elif style == "ethereal":
        prompt += "\nDesign soft, floating, dream-like movements."
    elif style == "minimal":
        prompt += "\nKeep it clean and simple, focusing on essential movement."
    elif style == "chaotic":
        prompt += "\nExplore complex, unpredictable multi-behavior interactions."
    
    # Simple performance guidance
    if performance_target == "fast":
        prompt += "\nOptimize for speed - keep particle counts moderate."
    elif performance_target == "quality":
        prompt += "\nPrioritize visual richness - can use more particles and effects."
    
    prompt += "\n\nExplore interesting behavior combinations and let emergent patterns surprise you!"
    
    return prompt


def build_modification_prompt(config: MultiBehaviorSketchConfig,
                            modification_request: str) -> str:
    """Build a modification prompt with minimal prescription."""
    
    # Simple analysis of current state
    behavior_summary = config.get_behavior_summary()
    total_particles = config.get_total_particle_count()
    
    prompt = f"""
Current sketch: "{config.sketch_name}"
Description: {config.description}
Current behaviors: {behavior_summary}
Total particles: {total_particles}

Modification request: {modification_request}

Enhance this sketch thoughtfully while preserving what makes it special.
"""
    
    return prompt


def build_creative_prompt(theme: str,
                        inspiration: Optional[str] = None,
                        mood: Optional[str] = None) -> str:
    """Build a creative prompt for artistic exploration."""
    
    prompt = f"Create an artistic Tölvera sketch exploring: {theme}"
    
    if inspiration:
        prompt += f"\nDraw inspiration from: {inspiration}"
    
    if mood:
        prompt += f"\nCapture the feeling of: {mood}"
    
    prompt += "\n\nLet your creativity flow and discover unexpected behavior combinations!"
    
    return prompt


def _analyze_current_config(config: MultiBehaviorSketchConfig) -> str:
    """Simple analysis without over-engineering."""
    
    analysis = []
    analysis.append(f"Behaviors: {config.get_behavior_summary()}")
    analysis.append(f"Total particles: {config.get_total_particle_count()}")
    analysis.append(f"Species: {config.get_max_species_index() + 1}")
    
    return " | ".join(analysis)