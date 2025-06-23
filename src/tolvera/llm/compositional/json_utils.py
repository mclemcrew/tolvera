# src/tolvera/llm/compositional/json_utils.py
"""
JSON cleanup and parsing utilities for the MoE system.
These handle common issues with LLM-generated JSON responses.
"""

import json
import re
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

def clean_json_response(response: str) -> str:
    """
    Clean up common JSON issues in LLM responses.
    
    Args:
        response: Raw response from LLM
        
    Returns:
        Cleaned JSON string
    """
    if not response:
        return "{}"
    
    # Remove common prefixes/suffixes
    response = response.strip()
    
    # Remove markdown code blocks
    if "```json" in response:
        response = response.split("```json")[1].split("```")[0].strip()
    elif "```" in response:
        # Find the first ``` and take content until next ```
        parts = response.split("```")
        if len(parts) >= 3:
            response = parts[1].strip()
    
    # Remove common text around JSON
    prefixes_to_remove = [
        "Here is the JSON:",
        "Here's the JSON:",
        "The JSON response is:",
        "Response:",
        "JSON:",
        "Here is",
        "Here's",
        "The plan is:",
        "The result is:",
    ]
    
    for prefix in prefixes_to_remove:
        if response.startswith(prefix):
            response = response[len(prefix):].strip()
            break
    
    # Find JSON boundaries - look for outermost braces
    start_idx = response.find('{')
    if start_idx == -1:
        # No opening brace found, return empty JSON
        return "{}"
    
    # Find matching closing brace
    brace_count = 0
    end_idx = len(response) - 1
    
    for i, char in enumerate(response[start_idx:], start_idx):
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                end_idx = i
                break
    
    response = response[start_idx:end_idx+1]
    
    # Fix common JSON issues
    response = fix_common_json_issues(response)
    
    return response.strip()

def fix_common_json_issues(json_str: str) -> str:
    """
    Fix common JSON formatting issues.
    
    Args:
        json_str: JSON string to fix
        
    Returns:
        Fixed JSON string
    """
    # Remove trailing commas before closing braces/brackets
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    
    # Fix single quotes to double quotes (common LLM mistake)
    # Be careful not to change quotes inside strings
    json_str = fix_quotes(json_str)
    
    # Remove comments (// style)
    json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
    
    # Remove extra whitespace
    json_str = re.sub(r'\s+', ' ', json_str)
    
    return json_str

def fix_quotes(json_str: str) -> str:
    """
    Fix quote issues in JSON string.
    Convert single quotes to double quotes while preserving string content.
    """
    result = []
    in_string = False
    escape_next = False
    i = 0
    
    while i < len(json_str):
        char = json_str[i]
        
        if escape_next:
            result.append(char)
            escape_next = False
        elif char == '\\':
            result.append(char)
            escape_next = True
        elif char == '"':
            result.append(char)
            in_string = not in_string
        elif char == "'" and not in_string:
            # Replace single quote with double quote outside of strings
            result.append('"')
        else:
            result.append(char)
        
        i += 1
    
    return ''.join(result)

def safe_json_parse(json_str: str, expected_type: type = dict) -> Optional[Dict[str, Any]]:
    """
    Safely parse JSON string with error handling.
    
    Args:
        json_str: JSON string to parse
        expected_type: Expected type of the parsed result
        
    Returns:
        Parsed JSON object or None if parsing fails
    """
    try:
        # First, clean the response
        cleaned = clean_json_response(json_str)
        logger.debug(f"Cleaned JSON: {cleaned}")
        
        # Try to parse
        parsed = json.loads(cleaned)
        
        # Validate type
        if not isinstance(parsed, expected_type):
            logger.warning(f"Parsed JSON is {type(parsed)}, expected {expected_type}")
            return None
        
        return parsed
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        logger.error(f"Attempted to parse: {repr(json_str[:200])}...")
        return None
    except Exception as e:
        logger.error(f"Unexpected error parsing JSON: {e}")
        return None

def extract_json_from_text(text: str) -> Optional[str]:
    """
    Extract the first JSON object from arbitrary text.
    
    Args:
        text: Text that may contain JSON
        
    Returns:
        Extracted JSON string or None if not found
    """
    # Look for JSON-like patterns
    patterns = [
        r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Simple nested braces
        r'\{.*?\}',  # Any content between braces
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            # Try to parse each match
            parsed = safe_json_parse(match)
            if parsed is not None:
                return match
    
    return None

def validate_tool_call_json(json_obj: Dict[str, Any]) -> bool:
    """
    Validate that a JSON object has the expected structure for tool calls.
    
    Args:
        json_obj: Parsed JSON object
        
    Returns:
        True if valid tool call structure
    """
    required_keys = {"tool_calls", "explanation"}
    
    if not isinstance(json_obj, dict):
        return False
    
    if not all(key in json_obj for key in required_keys):
        logger.warning(f"Missing required keys. Expected {required_keys}, got {json_obj.keys()}")
        return False
    
    if not isinstance(json_obj["tool_calls"], list):
        logger.warning(f"tool_calls should be list, got {type(json_obj['tool_calls'])}")
        return False
    
    if not isinstance(json_obj["explanation"], str):
        logger.warning(f"explanation should be string, got {type(json_obj['explanation'])}")
        return False
    
    # Validate each tool call
    for i, tool_call in enumerate(json_obj["tool_calls"]):
        if not isinstance(tool_call, dict):
            logger.warning(f"tool_call {i} should be dict, got {type(tool_call)}")
            return False
        
        if "tool_name" not in tool_call or "parameters" not in tool_call:
            logger.warning(f"tool_call {i} missing required keys")
            return False
    
    return True

def validate_task_plan_json(json_obj: Dict[str, Any]) -> bool:
    """
    Validate that a JSON object has the expected structure for task plans.
    
    Args:
        json_obj: Parsed JSON object
        
    Returns:
        True if valid task plan structure
    """
    required_keys = {"description", "steps"}
    
    if not isinstance(json_obj, dict):
        return False
    
    if not all(key in json_obj for key in required_keys):
        logger.warning(f"Missing required keys. Expected {required_keys}, got {json_obj.keys()}")
        return False
    
    if not isinstance(json_obj["description"], str):
        logger.warning(f"description should be string, got {type(json_obj['description'])}")
        return False
    
    if not isinstance(json_obj["steps"], list):
        logger.warning(f"steps should be list, got {type(json_obj['steps'])}")
        return False
    
    # Validate each step is a string
    for i, step in enumerate(json_obj["steps"]):
        if not isinstance(step, str):
            logger.warning(f"step {i} should be string, got {type(step)}")
            return False
    
    return True

# Testing function
def test_json_utils():
    """Test the JSON utilities with various malformed inputs."""
    print("🧪 Testing JSON Utilities")
    print("-" * 30)
    
    test_cases = [
        # Basic valid JSON
        '{"description": "test", "steps": ["step1"]}',
        
        # Markdown wrapped
        '```json\n{"description": "test", "steps": ["step1"]}\n```',
        
        # With prefix text
        'Here is the JSON: {"description": "test", "steps": ["step1"]}',
        
        # Single quotes
        "{'description': 'test', 'steps': ['step1']}",
        
        # Trailing comma
        '{"description": "test", "steps": ["step1",]}',
        
        # Mixed content
        'The plan is: {"description": "test", "steps": ["step1"]} Hope this helps!',
        
        # Tool call format
        '{"tool_calls": [{"tool_name": "create_particles", "parameters": {"n": 2}}], "explanation": "Created particles"}',
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest {i+1}: {repr(test_case[:50])}...")
        
        cleaned = clean_json_response(test_case)
        parsed = safe_json_parse(test_case)
        
        if parsed:
            print(f"✅ Successfully parsed: {list(parsed.keys())}")
        else:
            print(f"❌ Failed to parse")
            print(f"   Cleaned: {repr(cleaned[:100])}")

if __name__ == "__main__":
    test_json_utils()