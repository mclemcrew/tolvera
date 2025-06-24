# src/tolvera/llm/compositional/json_utils.py
"""
IMPROVED JSON cleanup and parsing utilities for the MoE system.
These handle common issues with LLM-generated JSON responses with enhanced robustness.
"""

import json
import re
import logging
from typing import Any, Dict, Optional, List

logger = logging.getLogger(__name__)

def clean_json_response(response: str) -> str:
    """
    IMPROVED: Clean up common JSON issues in LLM responses.
    Enhanced for robust MoE system with better JavaScript detection.
    """
    if not response:
        return "{}"
    
    response = response.strip()
    
    # Remove markdown code blocks
    if "```json" in response:
        response = response.split("```json")[1].split("```")[0].strip()
    elif "```" in response:
        parts = response.split("```")
        if len(parts) >= 3:
            response = parts[1].strip()
    
    # Remove common text prefixes
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
        "Here's the tool call:",
        "Tool call:",
        "Result:",
    ]
    
    for prefix in prefixes_to_remove:
        if response.startswith(prefix):
            response = response[len(prefix):].strip()
            break
    
    # Find JSON boundaries with better handling of strings and escapes
    start_idx = response.find('{')
    if start_idx == -1:
        return "{}"
    
    # Find matching closing brace with string awareness
    brace_count = 0
    end_idx = len(response) - 1
    in_string = False
    escape_next = False
    
    for i, char in enumerate(response[start_idx:], start_idx):
        if escape_next:
            escape_next = False
            continue
            
        if char == '\\' and in_string:
            escape_next = True
            continue
            
        if char == '"':
            in_string = not in_string
            continue
            
        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i
                    break
    
    response = response[start_idx:end_idx+1]
    
    # IMPROVED: Fix common JSON issues
    response = fix_common_json_issues(response)
    
    return response.strip()

def fix_common_json_issues(json_str: str) -> str:
    """
    IMPROVED: Fix common JSON formatting issues from LLMs with enhanced detection.
    """
    # Remove trailing commas
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    
    # Fix single quotes to double quotes
    json_str = fix_quotes(json_str)
    
    # Remove comments
    json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
    
    # IMPROVED: Enhanced JavaScript-like syntax fixes
    # Fix Math.random() and similar JavaScript code
    json_str = re.sub(r'Math\.random\(\)\s*\*\s*(\d+)', r'0.5', json_str)
    json_str = re.sub(r'Math\.random\(\)', r'0.5', json_str)
    json_str = re.sub(r'Math\.floor\([^)]+\)', r'0', json_str)
    json_str = re.sub(r'Math\.ceil\([^)]+\)', r'1', json_str)
    
    # Fix undefined/null values
    json_str = re.sub(r'\bundefined\b', 'null', json_str)
    
    # Fix missing quotes around property names
    json_str = re.sub(r'(\w+):', r'"\1":', json_str)
    
    # IMPROVED: Fix tool name normalization issues
    # Fix common tool name variations
    json_str = re.sub(r'"create_particle"', r'"create_particles"', json_str)
    json_str = re.sub(r'"set_particle_colour"', r'"set_species_color"', json_str)
    json_str = re.sub(r'"set_particle_color"', r'"set_species_color"', json_str)
    
    # Remove extra whitespace
    json_str = re.sub(r'\s+', ' ', json_str)
    
    return json_str

def fix_quotes(json_str: str) -> str:
    """
    IMPROVED: Fix quote issues in JSON string with better handling.
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
    IMPROVED: Safely parse JSON string with aggressive cleaning and error handling.
    Enhanced for robust MoE system with multiple fallback strategies.
    """
    if not json_str or not json_str.strip():
        logger.warning("Empty JSON string")
        return None
        
    try:
        # First, clean the response aggressively
        cleaned = clean_json_response(json_str)
        logger.debug(f"Cleaned JSON: {cleaned}")
        
        # Try multiple parsing strategies
        strategies = [
            lambda s: json.loads(s),  # Direct parse
            lambda s: json.loads(s.strip('"')),  # Remove outer quotes
            lambda s: json.loads(s.replace("'", '"')),  # Fix quotes again
            lambda s: json.loads(s.replace('\n', '').replace('\t', '')),  # Remove whitespace
            lambda s: json.loads(re.sub(r',\s*([}\]])', r'\1', s)),  # Remove trailing commas more aggressively
            lambda s: json.loads(extract_json_from_text(s) or "{}"),  # Extract JSON from text
            lambda s: json.loads(try_repair_truncated_json(s)),  # Try to repair truncated JSON
        ]
        
        parsed = None
        for i, strategy in enumerate(strategies):
            try:
                parsed = strategy(cleaned)
                if i > 0:
                    logger.info(f"JSON parsed successfully with strategy {i}")
                break
            except (json.JSONDecodeError, TypeError):
                continue
        
        if parsed is None:
            logger.error("All parsing strategies failed")
            return None
        
        # Validate type
        if not isinstance(parsed, expected_type):
            logger.warning(f"Parsed JSON is {type(parsed)}, expected {expected_type}")
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
    Enhanced to handle code snippets without recursion.
    
    Args:
        text: Text that may contain JSON
        
    Returns:
        Extracted JSON string or None if not found
    """
    if not text:
        return None
        
    # Find the JSON boundaries manually to avoid recursion
    start_idx = text.find('{')
    if start_idx == -1:
        return None
    
    # Count braces to find the matching end
    brace_count = 0
    in_string = False
    escape_next = False
    end_idx = len(text) - 1
    
    for i in range(start_idx, len(text)):
        char = text[i]
        
        if escape_next:
            escape_next = False
            continue
            
        if char == '\\' and in_string:
            escape_next = True
            continue
            
        if char == '"':
            in_string = not in_string
            continue
            
        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i
                    break
    
    if brace_count == 0:
        return text[start_idx:end_idx+1]
    
    return None

def try_repair_truncated_json(json_str: str) -> str:
    """
    Try to repair truncated JSON by adding missing closing brackets.
    This is a last-resort repair for JSON that got cut off.
    """
    if not json_str or not json_str.strip():
        return "{}"
        
    json_str = json_str.strip()
    
    # If it doesn't end with }, add missing closing braces
    if not json_str.endswith('}'):
        # Count opening vs closing braces
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        missing_braces = open_braces - close_braces
        
        # Count opening vs closing brackets  
        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')
        missing_brackets = open_brackets - close_brackets
        
        # Add missing closing brackets first, then braces
        repair = json_str
        if missing_brackets > 0:
            repair += ']' * missing_brackets
        if missing_braces > 0:
            repair += '}' * missing_braces
            
        logger.info(f"Repaired truncated JSON: added {missing_brackets} ] and {missing_braces} }}")
        return repair
    
    return json_str

def validate_task_plan_json(json_obj: Dict[str, Any]) -> bool:
    """
    IMPROVED: Validate that a JSON object has the expected structure for task plans.
    Enhanced for robust MoE system with better error handling.
    """
    if not isinstance(json_obj, dict):
        logger.warning(f"Expected dict, got {type(json_obj)}")
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
    
    # IMPROVED: More flexible step validation
    cleaned_steps = []
    for i, step in enumerate(json_obj["steps"]):
        if isinstance(step, str):
            cleaned_steps.append(step)
        elif isinstance(step, dict):
            # Convert dict to string
            if "step" in step:
                cleaned_steps.append(step["step"])
            elif "description" in step:
                cleaned_steps.append(step["description"])
            elif "action" in step:
                cleaned_steps.append(step["action"])
            else:
                cleaned_steps.append(str(step))
            logger.info(f"Converted step {i} from dict to string")
        else:
            cleaned_steps.append(str(step))
            logger.warning(f"Converted step {i} from {type(step)} to string")
    
    # Update with cleaned steps
    json_obj["steps"] = cleaned_steps
    
    return True

def validate_tool_call_json(json_obj: Dict[str, Any]) -> bool:
    """
    IMPROVED: Validate that a JSON object has the expected structure for tool calls.
    Enhanced for robust MoE system with tool name normalization.
    """
    if not isinstance(json_obj, dict):
        logger.warning(f"Expected dict, got {type(json_obj)}")
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
    
    # IMPROVED: Validate and normalize tool calls
    normalized_tool_calls = []
    for i, tool_call in enumerate(json_obj["tool_calls"]):
        if not isinstance(tool_call, dict):
            logger.warning(f"Tool call {i} should be dict, got {type(tool_call)}")
            continue
            
        if "tool_name" not in tool_call:
            logger.warning(f"Tool call {i} missing tool_name")
            continue
            
        if "parameters" not in tool_call:
            logger.warning(f"Tool call {i} missing parameters")
            tool_call["parameters"] = {}
        
        # Normalize tool names to match TOOL_REGISTRY
        tool_name = tool_call["tool_name"]
        original_name = tool_name
        
        # Common tool name normalizations
        tool_name_map = {
            "create_particle": "create_particles",
            "create_particles_with_code": "create_particles", 
            "apply_particle_colors": "set_species_color",
            "set_particle_color": "set_species_color",
            "set_particle_colour": "set_species_color",
            "apply_particle_movement": "set_species_velocity",
            "apply_physics_behavior": "apply_bouncing_behavior",
        }
        
        if tool_name in tool_name_map:
            tool_call["tool_name"] = tool_name_map[tool_name]
            logger.info(f"Normalized tool name from {original_name} to {tool_call['tool_name']}")
        
        # Ensure parameters dict exists
        if "parameters" not in tool_call:
            tool_call["parameters"] = {}
            logger.info(f"Added missing parameters dict for tool {tool_call['tool_name']}")
        
        # Ensure code_snippet and explanation exist (for backwards compatibility)
        if "code_snippet" not in tool_call:
            tool_call["code_snippet"] = ""
        if "explanation" not in tool_call:
            tool_call["explanation"] = f"Applied {tool_call['tool_name']}"
        
        normalized_tool_calls.append(tool_call)
    
    json_obj["tool_calls"] = normalized_tool_calls
    return True

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
        
        # JavaScript code issue
        '{"tool_calls": [{"tool_name": "create_particle", "parameters": {"position": [Math.random() * 800, Math.random() * 600]}}], "explanation": "Created particles"}',
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

def test_improved_json_utils():
    """
    Test the improved JSON utilities with focus on MoE system issues.
    This function tests the enhanced robustness for the MoE system.
    """
    print("🧪 Testing IMPROVED JSON Utilities for MoE System")
    print("=" * 50)
    
    # Test cases focusing on MoE system specific issues
    moe_test_cases = [
        # Tool name normalization
        '{"tool_calls": [{"tool_name": "create_particle", "parameters": {"n": 3}}], "explanation": "Created particles"}',
        
        # JavaScript Math.random() issue
        '{"tool_calls": [{"tool_name": "create_particles", "parameters": {"position": [Math.random() * 800, Math.random() * 600]}}], "explanation": "Created particles"}',
        
        # Color parameter variations
        '{"tool_calls": [{"tool_name": "set_particle_color", "parameters": {"color": [1.0, 0.0, 0.0, 1.0]}}], "explanation": "Set color"}',
        
        # Missing parameters
        '{"tool_calls": [{"tool_name": "create_particles"}], "explanation": "Created particles"}',
        
        # Task plan with dict steps
        '{"description": "Test plan", "steps": [{"step": "Create particles"}, {"action": "Set colors"}]}',
        
        # Complex JSON with multiple issues
        '''Here's the tool call: {
            "tool_calls": [
                {
                    "tool_name": "create_particle",
                    "parameters": {
                        "n": 3,
                        "position": [Math.random() * 800, Math.random() * 600],
                    }
                }
            ],
            "explanation": "Created particles with random positions"
        }''',
    ]
    
    for i, test_case in enumerate(moe_test_cases):
        print(f"\nMoE Test {i+1}: {repr(test_case[:60])}...")
        
        cleaned = clean_json_response(test_case)
        parsed = safe_json_parse(test_case)
        
        if parsed:
            print(f"✅ Successfully parsed: {list(parsed.keys())}")
            
            # Test specific validations
            if "tool_calls" in parsed:
                if validate_tool_call_json(parsed):
                    print(f"   ✅ Tool call validation passed")
                    # Check for normalized tool names
                    for tc in parsed["tool_calls"]:
                        print(f"      Tool: {tc['tool_name']}")
                else:
                    print(f"   ❌ Tool call validation failed")
            
            if "steps" in parsed:
                if validate_task_plan_json(parsed):
                    print(f"   ✅ Task plan validation passed")
                    print(f"      Steps: {len(parsed['steps'])}")
                else:
                    print(f"   ❌ Task plan validation failed")
        else:
            print(f"❌ Failed to parse")
            print(f"   Cleaned: {repr(cleaned[:100])}")

if __name__ == "__main__":
    test_json_utils()
    print("\n" + "="*50)
    test_improved_json_utils()