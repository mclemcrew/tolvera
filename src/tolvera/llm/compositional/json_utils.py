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
    Safely parse JSON string with error handling and aggressive cleaning.
    """
    if not json_str or not json_str.strip():
        logger.warning("Empty JSON string")
        return None
        
    try:
        # First, clean the response more aggressively
        cleaned = clean_json_response(json_str)
        logger.debug(f"Cleaned JSON: {cleaned}")
        
        # Try multiple parsing strategies
        strategies = [
            lambda s: json.loads(s),  # Direct parse
            lambda s: json.loads(s.strip('"')),  # Remove outer quotes
            lambda s: json.loads(s.replace("'", '"')),  # Fix quotes
            lambda s: json.loads(s.replace('\n', '').replace('\t', '')),  # Remove whitespace
        ]
        
        parsed = None
        for i, strategy in enumerate(strategies):
            try:
                parsed = strategy(cleaned)
                if i > 0:
                    logger.info(f"JSON parsed successfully with strategy {i}")
                break
            except json.JSONDecodeError:
                continue
        
        if parsed is None:
            logger.error("All parsing strategies failed")
            return None
        
        # Validate type
        if not isinstance(parsed, expected_type):
            logger.warning(f"Parsed JSON is {type(parsed)}, expected {expected_type}")
            # Try to convert if possible
            if expected_type == dict and isinstance(parsed, str):
                # Maybe it's a string containing JSON
                try:
                    parsed = json.loads(parsed)
                except:
                    return None
        
        return parsed
        
    except Exception as e:
        logger.error(f"Unexpected error parsing JSON: {e}")
        logger.error(f"Input was: {repr(json_str[:200])}...")
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

def validate_task_plan_json(json_obj: Dict[str, Any]) -> bool:
    """
    Validate that a JSON object has the expected structure for task plans.
    More flexible validation that handles common model variations.
    """
    if not isinstance(json_obj, dict):
        return False
    
    required_keys = {"description", "steps"}
    
    if not all(key in json_obj for key in required_keys):
        logger.warning(f"Missing required keys. Expected {required_keys}, got {json_obj.keys()}")
        return False
    
    if not isinstance(json_obj["description"], str):
        logger.warning(f"description should be string, got {type(json_obj['description'])}")
        return False
    
    if not isinstance(json_obj["steps"], list):
        logger.warning(f"steps should be list, got {type(json_obj['steps'])}")
        return False
    
    # More flexible step validation - convert objects to strings if needed
    cleaned_steps = []
    for i, step in enumerate(json_obj["steps"]):
        if isinstance(step, str):
            cleaned_steps.append(step)
        elif isinstance(step, dict):
            # Convert dict to string (common model error)
            if "step" in step:
                cleaned_steps.append(step["step"])
            elif "description" in step:
                cleaned_steps.append(step["description"])
            else:
                # Use the whole dict as string representation
                cleaned_steps.append(str(step))
            logger.info(f"Converted step {i} from dict to string")
        else:
            # Convert other types to string
            cleaned_steps.append(str(step))
            logger.warning(f"Converted step {i} from {type(step)} to string")
    
    # Update the original json_obj with cleaned steps
    json_obj["steps"] = cleaned_steps
    
    return True

def validate_tool_call_json(json_obj: Dict[str, Any]) -> bool:
    """
    Validate that a JSON object has the expected structure for tool calls.
    More flexible validation that handles common model variations.
    """
    if not isinstance(json_obj, dict):
        return False
    
    required_keys = {"tool_calls", "explanation"}
    
    if not all(key in json_obj for key in required_keys):
        logger.warning(f"Missing required keys. Expected {required_keys}, got {json_obj.keys()}")
        return False
    
    if not isinstance(json_obj["tool_calls"], list):
        logger.warning(f"tool_calls should be list, got {type(json_obj['tool_calls'])}")
        return False
    
    if not isinstance(json_obj["explanation"], str):
        logger.warning(f"explanation should be string, got {type(json_obj['explanation'])}")
        return False
    
    # Validate and clean each tool call
    cleaned_tool_calls = []
    for i, tool_call in enumerate(json_obj["tool_calls"]):
        if not isinstance(tool_call, dict):
            logger.warning(f"tool_call {i} should be dict, got {type(tool_call)}")
            continue
        
        # Ensure required keys exist
        if "tool_name" not in tool_call:
            logger.warning(f"tool_call {i} missing tool_name")
            continue
            
        if "parameters" not in tool_call:
            # Add empty parameters if missing
            tool_call["parameters"] = {}
            logger.info(f"Added empty parameters to tool_call {i}")
        
        cleaned_tool_calls.append(tool_call)
    
    # Update with cleaned tool calls
    json_obj["tool_calls"] = cleaned_tool_calls
    
    return len(cleaned_tool_calls) > 0 or len(json_obj["tool_calls"]) == 0  # Allow empty tool_calls


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