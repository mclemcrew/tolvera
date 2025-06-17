"""
Generation prompts for Tölvera sketch creation - Simplified approach.

Focused on organic discovery rather than prescriptive guidance.
"""

from .system_prompts import (
    GENERATION_SYSTEM_PROMPT,
    MODIFICATION_SYSTEM_PROMPT
)

from .prompt_builders import (
    build_generation_prompt,
    build_modification_prompt,
    build_creative_prompt
)

__all__ = [
    # System prompts
    "GENERATION_SYSTEM_PROMPT",
    "MODIFICATION_SYSTEM_PROMPT",
    
    # Prompt builders
    "build_generation_prompt",
    "build_modification_prompt", 
    "build_creative_prompt"
]