"""
Template system for Tölvera code generation.

Clean imports for template functionality.
"""

from .builtin_templates import (
    get_builtin_templates,
    get_template_info
)

__all__ = [
    "get_builtin_templates",
    "get_template_info"
]