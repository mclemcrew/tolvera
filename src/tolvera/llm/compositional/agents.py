import logging
from typing import List
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from .tools import TaskResult, TaskPlan, GeneratedScript, ToolCall
from tolvera.llm.compositional.prompts import conductor_agent_prompt, particle_agent_prompt, color_agent_prompt, physics_agent_prompt, motion_agent_prompt, composition_agent_prompt

logger = logging.getLogger(__name__)

""" 
List of all the agents in this file:
- Base Agent
- Conductor Agent
- Particle Agent
- Color Agent
- Physics Agent
- Motion Agent
"""


class BaseAgent:
    def __init__(self, model_name: str = "qwen2.5:3b"):
        self.model_name = model_name
        self.model = OpenAIModel(
            model_name=model_name,
            provider=OpenAIProvider(
                base_url="http://localhost:11434/v1",
                api_key="ollama"
            )
        )
        
        self.model_settings = ModelSettings(
            max_tokens=10000,  
            timeout=60,     
            temperature=0.1 
        )
        logger.debug(f"Initialized {self.__class__.__name__} with model {model_name} (max_tokens=10000)")

class ConductorAgent(BaseAgent):
    def __init__(self):
        super().__init__("qwen2.5:3b")
        
        self.system_prompt_text = conductor_agent_prompt.CONDUCTOR_AGENT_PROMPT
        self.agent = Agent(
            model=self.model,
            output_type=TaskPlan,
            system_prompt=self.system_prompt_text,
            model_settings=self.model_settings
        )


    async def plan_task(self, user_request: str) -> TaskPlan:
        # I chose four steps, but it could really be anything here.
        prompt = f'Create a 4-step plan for: "{user_request}"\n\nExtract specific parameters from the request and include them in your steps.'
        
        n_particles = self._extract_particle_count_from_request(user_request)
        color_name, _ = self._extract_color_from_request(user_request)
        behavior = self._extract_behavior_from_request(user_request)
        
        logger.debug(f"Extracted params: {n_particles} {color_name} particles with {behavior}")
        
        try:
            logger.debug(f"ConductorAgent: Sending prompt to LLM: {prompt}")
            result = await self.agent.run(prompt)
            plan = result.output
            logger.info(f"Conductor planned: {plan.description}")
            return plan
        except Exception as e:
            logger.error(f"ConductorAgent failed: {e}")
            
            # Try to extract raw response from pydantic-ai exception
            raw_response = None
            if hasattr(e, '_raw_response'):
                raw_response = e._raw_response
            elif hasattr(e, 'response'):
                raw_response = e.response
            elif hasattr(e, 'args') and e.args:
                # Sometimes the response is in the args...this isn't fun to deal with
                for arg in e.args:
                    if isinstance(arg, str) and len(arg) > 10:
                        raw_response = arg
                        break
            
            if raw_response:
                logger.error(f"Raw LLM response: {raw_response}")
            else:
                logger.error(f"No raw response available. Exception type: {type(e)}")
                logger.error(f"Exception details: {str(e)}")
                logger.error(f"Exception dir: {[attr for attr in dir(e) if not attr.startswith('_')]}")
            
            raise Exception(f"ConductorAgent failed to generate valid JSON. Raw error: {e}")
    
    def _extract_particle_count_from_request(self, request: str) -> int:
        import re
        
        # This is just beacuse this kept happeneing over and over again.
        number_words = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
            "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
            "a": 1, "single": 1, "couple": 2, "few": 3, "several": 5,
            "some": 4, "bunch": 8, "many": 12, "lots": 15, "loads": 18,
            "tons": 25, "hundreds": 50  # Cap large numbers reasonably 
        }
        
        request_lower = request.lower()
        
        # Check for number words
        for word, num in number_words.items():
            if f" {word} " in f" {request_lower} " or request_lower.startswith(word):
                return num
        
        # Check for digits
        numbers = re.findall(r'\b(\d+)\b', request)
        if numbers:
            num = int(numbers[0])
            # Cap extremely large numbers for performance
            return max(1, min(100, num))
        
        # Handle ranges like "between 5 and 10"
        range_match = re.search(r'between\s+(\d+)\s+and\s+(\d+)', request_lower)
        if range_match:
            start, end = int(range_match.group(1)), int(range_match.group(2))
            return min(max(start, end), 50)  # Use larger number, cap at 50
        
        # Context-aware defaults based on complexity
        if any(word in request_lower for word in ["simple", "basic", "quick"]):
            return 1
        elif any(word in request_lower for word in ["complex", "swarm", "crowd", "field"]):
            return 15
        else:
            return 3  # Safe default
    
    def _extract_color_from_request(self, request: str) -> tuple:
        # Easy lookup for this
        color_map = {
            "red": ([1.0, 0.0, 0.0, 1.0], "red"),
            "green": ([0.0, 1.0, 0.0, 1.0], "green"),
            "blue": ([0.0, 0.0, 1.0, 1.0], "blue"),
            "yellow": ([1.0, 1.0, 0.0, 1.0], "yellow"),
            "orange": ([1.0, 0.5, 0.0, 1.0], "orange"),
            "purple": ([0.5, 0.0, 1.0, 1.0], "purple"),
            "pink": ([1.0, 0.5, 0.8, 1.0], "pink"),
            "white": ([1.0, 1.0, 1.0, 1.0], "white"),
            "black": ([0.0, 0.0, 0.0, 1.0], "black"),
            "cyan": ([0.0, 1.0, 1.0, 1.0], "cyan"),
            "magenta": ([1.0, 0.0, 1.0, 1.0], "magenta")
        }
        
        request_lower = request.lower()
        for color_name, (rgba, name) in color_map.items():
            if color_name in request_lower:
                return name, rgba
        
        return "green", [0.0, 1.0, 0.0, 1.0]  # Default
    
    def _extract_behavior_from_request(self, request: str) -> str:
        request_lower = request.lower()
        
        if "bounce" in request_lower:
            return "bouncing physics with collision detection"
        elif "spiral" in request_lower:
            return "spiral motion pattern"
        elif "circle" in request_lower and "moving" in request_lower:
            return "circular motion pattern"
        elif "right" in request_lower:
            return "rightward movement"
        elif "flock" in request_lower:
            return "flocking behavior"
        else:
            return "appropriate physics behavior"

class ParticleCreationAgent(BaseAgent):
    def __init__(self):
        super().__init__("qwen2.5:3b")
        
        self.system_prompt_text = particle_agent_prompt.PARTICLE_AGENT_PROMPT

        self.agent = Agent(
            model=self.model,
            output_type=TaskResult,
            system_prompt=self.system_prompt_text,
            model_settings=self.model_settings
        )


    async def execute_task(self, task: str, user_request: str = None) -> TaskResult:
        
        # Use original user request for parameter extraction if available
        extraction_source = user_request if user_request else task
        prompt = f'User Request: "{user_request}"\nStep: "{task}"\n\nExtract the exact particle count from the user request.'
        
        # Extract parameters for logging
        n_particles = self._extract_number_from_request(extraction_source)
        shape = self._extract_shape_from_request(extraction_source)
        speed_value = self._extract_speed_from_request(extraction_source)
        
        logger.debug(f"Extracted params: {n_particles} {shape} particles, speed {speed_value}")
        
        try:
            result = await self.agent.run(prompt)
            task_result = result.output
            logger.info(f"✨ ParticleCreationAgent: {task_result.explanation}")
            return task_result
        except Exception as e:
            logger.error(f"ParticleCreationAgent failed: {e}")
            
            # Try to extract raw response from pydantic-ai exception
            raw_response = None
            if hasattr(e, '_raw_response'):
                raw_response = e._raw_response
            elif hasattr(e, 'response'):
                raw_response = e.response
            elif hasattr(e, 'args') and e.args:
                # Sometimes the response is in the args
                for arg in e.args:
                    if isinstance(arg, str) and len(arg) > 10:
                        raw_response = arg
                        break
            
            if raw_response:
                logger.error(f"Raw LLM response: {raw_response}")
            else:
                logger.error(f"No raw response available. Exception type: {type(e)}")
                logger.error(f"Exception details: {str(e)}")
                logger.error(f"Exception dir: {[attr for attr in dir(e) if not attr.startswith('_')]}")
            
            raise Exception(f"ParticleCreationAgent failed to generate valid JSON. Raw error: {e}")
    
    def _extract_number_from_request(self, request: str) -> int:
        import re
        
        number_words = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
            "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
            "a": 1, "single": 1, "couple": 2, "pair": 2,
            "few": 3, "several": 5, "some": 4, "handful": 4,
            "bunch": 8, "many": 12, "lots": 15, "loads": 18,
            "tons": 25, "hundreds": 50, "multitude": 30,
            "swarm": 20, "crowd": 25, "horde": 35
        }
        
        request_lower = request.lower()
        
        for word, num in number_words.items():
            if f" {word} " in f" {request_lower} " or request_lower.startswith(word):
                return num
        
        numbers = re.findall(r'\b(\d+)\b', request)
        if numbers:
            num = int(numbers[0])
            # Cap extremely large numbers for performance
            return max(1, min(100, num))
        
        range_patterns = [
            r'between\s+(\d+)\s+and\s+(\d+)',
            r'(\d+)\s+to\s+(\d+)',
            r'(\d+)\s*-\s*(\d+)'
        ]
        
        for pattern in range_patterns:
            range_match = re.search(pattern, request_lower)
            if range_match:
                start, end = int(range_match.group(1)), int(range_match.group(2))
                return min(max(start, end), 100)
        
        if any(word in request_lower for word in ["simple", "basic", "single", "one"]):
            return 1
        elif any(word in request_lower for word in ["complex", "elaborate", "intricate"]):
            return 8
        elif any(word in request_lower for word in ["field", "cloud", "galaxy", "universe"]):
            return 25
        
        singular_words = ["particle", "circle", "point", "dot", "pixel"]
        plural_words = ["particles", "circles", "points", "dots", "pixels"]
        
        has_singular = any(word in request_lower for word in singular_words)
        has_plural = any(word in request_lower for word in plural_words)
        
        if has_singular and not has_plural:
            return 1
        elif has_plural or has_singular: 
            return 5
        
        # If nothing else works...
        return 3
    
    def _extract_shape_from_request(self, request: str) -> str:
        request_lower = request.lower()
        
        # Shape mapping from user words to Tölvera shape names
        shape_mappings = {
            "square": "rect",
            "squares": "rect", 
            "rect": "rect",
            "rectangle": "rect",
            "rectangles": "rect",
            "circle": "circle",
            "circles": "circle",
            "dot": "circle",
            "dots": "circle",
            "triangle": "triangle",
            "triangles": "triangle",
            "point": "point",
            "points": "point",
            "line": "line",
            "lines": "line"
        }
        
        # Search for shape words in the request
        for word, tolvera_shape in shape_mappings.items():
            if word in request_lower:
                return tolvera_shape
        
        # Default to circle if no shape specified
        return "circle"
    
    def _extract_speed_from_request(self, request: str) -> float:
        # This was a last minute addition and isn't working too well...
        #TODO
        request_lower = request.lower()
        
        # Check for speed descriptors
        if "very rapidly" in request_lower or "extremely fast" in request_lower:
            return 3.0
        elif "rapidly" in request_lower or "very fast" in request_lower:
            return 2.5
        elif "quickly" in request_lower or "fast" in request_lower:
            return 2.0
        elif "slowly" in request_lower:
            return 0.5
        elif "very slowly" in request_lower:
            return 0.3
        else:
            return 1.0  # Default speed

class ColorAgent(BaseAgent):
    def __init__(self):
        super().__init__("qwen2.5:3b")
        
        self.system_prompt_text = color_agent_prompt.COLOR_AGENT_PROMPT

        self.agent = Agent(
            model=self.model,
            output_type=TaskResult,
            system_prompt=self.system_prompt_text,
            model_settings=self.model_settings
        )


    async def execute_task(self, task: str, user_request: str = None) -> TaskResult:
        # Use original user request for color extraction if available
        extraction_source = user_request if user_request else task
        prompt = f'User Request: "{user_request}"\nStep: "{task}"\n\nExtract the exact color from the user request.'
        
        # Extract color for logging
        color, color_name = self._extract_color_from_request(extraction_source)
        
        logger.debug(f"Extracted color: {color_name} {color}")
        
        try:
            result = await self.agent.run(prompt)
            task_result = result.output
            logger.info(f"🎨 ColorAgent: {task_result.explanation}")
            return task_result
        except Exception as e:
            logger.error(f"ColorAgent failed: {e}")
            
            # Try to extract raw response from pydantic-ai exception
            raw_response = None
            if hasattr(e, '_raw_response'):
                raw_response = e._raw_response
            elif hasattr(e, 'response'):
                raw_response = e.response
            elif hasattr(e, 'args') and e.args:
                # Sometimes the response is in the args
                for arg in e.args:
                    if isinstance(arg, str) and len(arg) > 10:
                        raw_response = arg
                        break
            
            if raw_response:
                logger.error(f"Raw LLM response: {raw_response}")
            else:
                logger.error(f"No raw response available. Exception type: {type(e)}")
                logger.error(f"Exception details: {str(e)}")
                logger.error(f"Exception dir: {[attr for attr in dir(e) if not attr.startswith('_')]}")
            
            raise Exception(f"ColorAgent failed to generate valid JSON. Raw error: {e}")
    
    def _extract_color_from_request(self, request: str) -> tuple:
        # Common colors for fast lookup
        color_map = {
            "red": [1.0, 0.0, 0.0, 1.0],
            "green": [0.0, 1.0, 0.0, 1.0],
            "blue": [0.0, 0.0, 1.0, 1.0],
            "yellow": [1.0, 1.0, 0.0, 1.0],
            "orange": [1.0, 0.5, 0.0, 1.0],
            "purple": [0.5, 0.0, 1.0, 1.0],
            "pink": [1.0, 0.5, 0.8, 1.0],
            "white": [1.0, 1.0, 1.0, 1.0],
            "black": [0.0, 0.0, 0.0, 1.0],
            "brown": [0.6, 0.3, 0.1, 1.0],
            "gray": [0.5, 0.5, 0.5, 1.0],
            "grey": [0.5, 0.5, 0.5, 1.0],
            "cyan": [0.0, 1.0, 1.0, 1.0],
            "magenta": [1.0, 0.0, 1.0, 1.0],
            "lime": [0.5, 1.0, 0.0, 1.0],
            "navy": [0.0, 0.0, 0.5, 1.0],
            "maroon": [0.5, 0.0, 0.0, 1.0],
            "teal": [0.0, 0.5, 0.5, 1.0],
            "violet": [0.4, 0.0, 0.8, 1.0],
            "indigo": [0.3, 0.0, 0.5, 1.0],
            "turquoise": [0.2, 0.8, 0.8, 1.0],
            "coral": [1.0, 0.5, 0.3, 1.0],
            "salmon": [1.0, 0.6, 0.6, 1.0]
        }
        
        request_lower = request.lower()
        
        # Check for color modifiers
        is_bright = "bright" in request_lower
        is_dark = "dark" in request_lower
        is_light = "light" in request_lower
        
        # Find the base color in known colors
        base_color = None
        base_name = None
        
        for color_name, color_value in color_map.items():
            if color_name in request_lower:
                base_color = color_value.copy()
                base_name = color_name
                break
        
        # If color found in map, apply modifiers and return
        if base_color:
            if is_bright:
                base_color[0] = min(1.0, base_color[0] + 0.1)
                base_color[1] = min(1.0, base_color[1] + 0.1)
                base_color[2] = min(1.0, base_color[2] + 0.1)
                base_name = f"bright {base_name}"
            elif is_dark:
                base_color[0] = max(0.0, base_color[0] - 0.3)
                base_color[1] = max(0.0, base_color[1] - 0.3)
                base_color[2] = max(0.0, base_color[2] - 0.3)
                base_name = f"dark {base_name}"
            elif is_light:
                base_color[0] = min(1.0, base_color[0] + 0.3)
                base_color[1] = min(1.0, base_color[1] + 0.3)
                base_color[2] = min(1.0, base_color[2] + 0.3)
                base_name = f"light {base_name}"
            
            return base_color, base_name
        
        # If no known color found, try to extract unknown color and use LLM for it
        unknown_color = self._extract_unknown_color_name(request_lower)
        if unknown_color:
            logger.info(f"🎨 Unknown color '{unknown_color}' found, using LLM to convert to RGB")
            llm_color = self._convert_color_with_llm(unknown_color)
            if llm_color:
                return llm_color, unknown_color
            else:
                logger.warning(f"LLM failed to convert '{unknown_color}', using fallback")
                return [0.8, 0.2, 0.2, 1.0], unknown_color  # Reddish fallback
        
        # Default only if NO color mentioned at all
        return [0.0, 1.0, 0.0, 1.0], "green"
    
    def _extract_unknown_color_name(self, request_lower: str) -> str:
        import re

        color_indicators = ["color", "colored", "colour", "coloured"]
        for indicator in color_indicators:
            pattern = rf'(\w+)\s+(?:in\s+)?{indicator}'
            match = re.search(pattern, request_lower)
            if match:
                potential_color = match.group(1)
                # Skip if it's a known color or not actually a color word
                if potential_color not in ["red", "green", "blue", "yellow", "orange", "purple", "pink", "white", "black", "brown", "gray", "grey", "cyan", "magenta", "lime", "navy", "maroon", "teal", "violet", "indigo", "turquoise", "coral", "salmon"]:
                    return potential_color
        
        words = request_lower.split()
        for word in words:
            # Skip common non-color words and known colors
            if len(word) > 3 and word not in ["that", "are", "and", "the", "with", "particles", "circles", "around", "screen", "bounce", "move", "create", "three", "five", "red", "green", "blue", "yellow", "orange", "purple", "pink", "white", "black", "brown", "gray", "grey", "cyan", "magenta", "lime", "navy", "maroon", "teal", "violet", "indigo", "turquoise", "coral", "salmon"]:
                # unless maybe that word is actually a color?
                return word
        
        return None
    
    def _convert_color_with_llm(self, color_name: str) -> List[float]:
        """Use LLM to convert unknown color name to RGBA values."""
        try:
            import asyncio
            
            # Create a simple agent for color conversion
            color_conversion_agent = Agent(
                model=self.model,
                result_type=str,
                system_prompt=f"""You are a color expert. Convert color names to RGBA values.

Your job is to convert the color name '{color_name}' to precise RGBA values between 0.0 and 1.0.

CRITICAL RULES:
1. Output ONLY a JSON array like [r, g, b, 1.0]
2. All values must be between 0.0 and 1.0
3. Alpha should always be 1.0 for opaque colors
4. No text before or after the JSON array
5. Use your knowledge of standard color names

EXAMPLES:
- crimson → [0.86, 0.08, 0.24, 1.0]
- chartreuse → [0.5, 1.0, 0.0, 1.0]
- periwinkle → [0.8, 0.8, 1.0, 1.0]

Respond with ONLY the JSON array for '{color_name}'.""",
                model_settings=self.model_settings
            )
            
            # Run the conversion (we need to handle this synchronously so it slows it down unfortunately)
            # This is a bit of a hack but necessary since we're in a sync method
            try:
                import nest_asyncio
                nest_asyncio.apply()
            except:
                pass  # nest_asyncio might not be available
            
            # Create a sync wrapper
            def run_color_conversion():
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(asyncio.run, color_conversion_agent.run(f"Convert '{color_name}' to RGBA"))
                            result = future.result(timeout=60)
                    else:
                        result = loop.run_until_complete(color_conversion_agent.run(f"Convert '{color_name}' to RGBA"))
                except:
                    result = asyncio.run(color_conversion_agent.run(f"Convert '{color_name}' to RGBA"))
                
                return result.output if hasattr(result, 'output') else str(result)
            
            response = run_color_conversion()
            logger.debug(f"🎨 LLM color conversion response: {response}")
            
            # Parse the response
            cleaned_response = response.strip()
            
            # Try to extract JSON array from response
            # TODO.... I'm just realizing that I changed the way we were handling this so I'm not certain if we should be doing this anymore???
            import json
            try:
                color_values = json.loads(cleaned_response)
                if isinstance(color_values, list) and len(color_values) >= 3:
                    # Make sure we have 4 values (RGBA)
                    if len(color_values) == 3:
                        color_values.append(1.0)  # Add alpha if not included
                    
                    rgba = [max(0.0, min(1.0, float(v))) for v in color_values[:4]]
                    logger.info(f"🎨 Successfully converted '{color_name}' to {rgba}")
                    return rgba
            except json.JSONDecodeError:
                # Try to extract numbers from response
                import re
                numbers = re.findall(r'\d*\.?\d+', cleaned_response)
                if len(numbers) >= 3:
                    rgba = [max(0.0, min(1.0, float(n))) for n in numbers[:3]]
                    rgba.append(1.0)  # Add alpha
                    logger.info(f"🎨 Extracted RGB from response for '{color_name}': {rgba}")
                    return rgba
        
        except Exception as e:
            logger.warning(f"Error in LLM color conversion for '{color_name}': {e}")
        
        return None
    

class MotionAgent(BaseAgent):
    def __init__(self):
        super().__init__("qwen2.5:3b")
        
        self.system_prompt_text = motion_agent_prompt.MOTION_AGENT_PROMPT

        self.agent = Agent(
            model=self.model,
            output_type=TaskResult,
            system_prompt=self.system_prompt_text,
            model_settings=self.model_settings
        )


    async def execute_task(self, task: str, user_request: str = None) -> TaskResult:
        # Use original user request for motion extraction if available
        extraction_source = user_request if user_request else task
        prompt = f'User Request: "{user_request}"\nStep: "{task}"\n\nCreate motion pattern based on the user request. Be creative for complex motion descriptions.'
        
        # Extract motion parameters for logging
        velocity, motion_type = self._extract_motion_from_request(extraction_source)
        
        logger.debug(f"Extracted motion: {motion_type} motion with velocity {velocity}")
        
        try:
            result = await self.agent.run(prompt)
            task_result = result.output
            logger.info(f"🏃 MotionAgent: {task_result.explanation}")
            return task_result
        except Exception as e:
            logger.error(f"MotionAgent failed: {e}")
            
            # Try to extract raw response from pydantic-ai exception
            raw_response = None
            if hasattr(e, '_raw_response'):
                raw_response = e._raw_response
            elif hasattr(e, 'response'):
                raw_response = e.response
            elif hasattr(e, 'args') and e.args:
                # Sometimes the response is in the args
                for arg in e.args:
                    if isinstance(arg, str) and len(arg) > 10:
                        raw_response = arg
                        break
            
            if raw_response:
                logger.error(f"Raw LLM response: {raw_response}")
            else:
                logger.error(f"No raw response available. Exception type: {type(e)}")
                logger.error(f"Exception details: {str(e)}")
                logger.error(f"Exception dir: {[attr for attr in dir(e) if not attr.startswith('_')]}")
            
            raise Exception(f"MotionAgent failed to generate valid JSON. Raw error: {e}")
    
    def _extract_motion_from_request(self, request: str) -> tuple:
        request_lower = request.lower()
        
        # Complex motion patterns
        if "spiral" in request_lower:
            return [1.5, 0.0], "spiral"
        elif "figure" in request_lower and "8" in request_lower:
            return [1.0, 0.0], "figure-8"
        elif "orbit" in request_lower or "circle" in request_lower:
            return [2.0, 0.0], "circular"
        elif "wave" in request_lower or "sine" in request_lower:
            return [2.0, 0.0], "wave"
        elif "zigzag" in request_lower or "zig zag" in request_lower:
            return [1.5, 1.5], "zigzag"
        elif "random" in request_lower or "chaotic" in request_lower:
            return [1.0, 1.0], "random"
        elif "gentle" in request_lower or "float" in request_lower:
            return [0.5, 0.5], "gentle"
        
        # Basic directional movement
        elif "right" in request_lower:
            base_velocity = [2.0, 0.0]
            speed_mult = self._analyze_speed_from_request(request_lower)
            return [base_velocity[0] * speed_mult, base_velocity[1] * speed_mult], "rightward"
        elif "left" in request_lower:
            base_velocity = [-2.0, 0.0]
            speed_mult = self._analyze_speed_from_request(request_lower)
            return [base_velocity[0] * speed_mult, base_velocity[1] * speed_mult], "leftward"
        elif "up" in request_lower:
            base_velocity = [0.0, -2.0]
            speed_mult = self._analyze_speed_from_request(request_lower)
            return [base_velocity[0] * speed_mult, base_velocity[1] * speed_mult], "upward"
        elif "down" in request_lower:
            base_velocity = [0.0, 2.0]
            speed_mult = self._analyze_speed_from_request(request_lower)
            return [base_velocity[0] * speed_mult, base_velocity[1] * speed_mult], "downward"
        
        # Default with enhanced speed analysis
        base_velocity = [2.0, 0.0]
        speed_mult = self._analyze_speed_from_request(request_lower)
        return [base_velocity[0] * speed_mult, base_velocity[1] * speed_mult], "linear"
    
    def _analyze_speed_from_request(self, request: str) -> float:
        # Quick keyword detection for common cases (fast path)
        speed_keywords = {
            # Very fast descriptors
            "very rapidly": 3.0, "extremely fast": 4.0, "super fast": 3.5, "blazing": 4.5,
            "very quick": 2.8, "extremely rapid": 4.2, "super rapid": 3.8,
            "lightning": 5.0, "incredibly fast": 4.8, "ridiculously fast": 5.5,
            
            # Fast descriptors
            "rapidly": 2.5, "quickly": 2.0, "fast": 1.8, "quick": 1.7, "swift": 1.9,
            "speedy": 2.2, "brisk": 1.6, "hasty": 1.8,
            
            # Moderate descriptors
            "moderate": 1.0, "normal": 1.0, "regular": 1.0, "steady": 1.0,
            
            # Slow descriptors
            "slowly": 0.6, "slow": 0.5, "leisurely": 0.4, "gradual": 0.3,
            "gentle": 0.4, "calm": 0.5, "peaceful": 0.4,
            
            # Very slow descriptors
            "very slowly": 0.2, "extremely slow": 0.15, "super slow": 0.25,
            "incredibly slow": 0.1, "glacial": 0.05, "snail": 0.1,
        }
        
        # Check for exact keyword matches first
        for keyword, multiplier in speed_keywords.items():
            if keyword in request:
                logger.info(f"🏃 Speed keyword detected: '{keyword}' → {multiplier}x")
                return multiplier
        
        # Check for compound modifiers + base words
        modifiers = {
            "very": 1.8, "extremely": 2.5, "super": 2.0, "incredibly": 2.8,
            "quite": 1.3, "somewhat": 1.1, "rather": 1.2, "pretty": 1.4,
            "ultra": 3.0, "mega": 2.8, "hyper": 3.2
        }
        
        base_speed_words = {
            "fast": 1.8, "quick": 1.7, "rapid": 2.2, "swift": 1.9,
            "slow": 0.5, "gradual": 0.3, "gentle": 0.4
        }
        
        # Look for modifier + base word combinations
        for modifier, mod_mult in modifiers.items():
            if modifier in request:
                for base_word, base_mult in base_speed_words.items():
                    if base_word in request:
                        combined_mult = base_mult * mod_mult
                        logger.info(f"🏃 Speed combination detected: '{modifier} {base_word}' → {combined_mult}x")
                        return combined_mult
        
        # Check for single base words
        for word, multiplier in base_speed_words.items():
            if word in request:
                logger.info(f"🏃 Base speed word detected: '{word}' → {multiplier}x")
                return multiplier
        
        # If no keywords found, try LLM analysis for complex phrases
        llm_speed = self._analyze_speed_with_llm(request)
        if llm_speed is not None:
            logger.info(f"🧠 LLM speed analysis: → {llm_speed}x")
            return llm_speed
        
        # Default speed
        logger.info("🏃 Using default speed: 1.0x")
        return 1.0
    
    def _analyze_speed_with_llm(self, request: str) -> float:
        """Use LLM to analyze complex speed descriptors that keywords missed."""
        try:
            
            speed_analysis_agent = Agent(
                model=self.model,
                result_type=str,
                system_prompt=f"""You are a speed analysis expert. Analyze movement speed descriptors in natural language.

Your job is to convert speed descriptions to numeric multipliers relative to normal speed (1.0).

CRITICAL RULES:
1. Output ONLY a single number between 0.05 and 6.0
2. No text before or after the number
3. Use your understanding of natural language intensity
4. Normal/regular speed = 1.0

SPEED SCALE EXAMPLES:
- "glacially slow" → 0.05
- "very slowly" → 0.2  
- "slowly" → 0.5
- "normal" → 1.0
- "quickly" → 1.8
- "very rapidly" → 3.0
- "extremely fast" → 4.0
- "lightning fast" → 5.0
- "impossibly fast" → 6.0

Analyze the speed descriptor in: "{request}"

Respond with ONLY the numeric multiplier.""",
                model_settings=self.model_settings
            )
            
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're in async context, but need sync result
                    # Use the same pattern as the color agent
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, speed_analysis_agent.run(f"Analyze speed in: {request}"))
                        result = future.result(timeout=60)
                else:
                    result = loop.run_until_complete(speed_analysis_agent.run(f"Analyze speed in: {request}"))
            except:
                # Fallback: run in new event loop
                result = asyncio.run(speed_analysis_agent.run(f"Analyze speed in: {request}"))
            
            response = result.output if hasattr(result, 'output') else str(result)
            logger.debug(f"🧠 LLM speed analysis response: {response}")
            
            # Parse the response
            cleaned_response = response.strip()
            
            # Try to extract a number from the response
            import re
            numbers = re.findall(r'\d*\.?\d+', cleaned_response)
            if numbers:
                speed_mult = float(numbers[0])
                # Clamp to reasonable range
                speed_mult = max(0.05, min(6.0, speed_mult))
                return speed_mult
        
        except Exception as e:
            logger.warning(f"Error in LLM speed analysis: {e}")
        
        return None
    

class PhysicsAgent(BaseAgent):
    def __init__(self):
        super().__init__("qwen2.5:3b")
        
        self.system_prompt_text = physics_agent_prompt.PHYSICS_AGENT_PROMPT

        self.agent = Agent(
            model=self.model,
            output_type=TaskResult,
            system_prompt=self.system_prompt_text,
            model_settings=self.model_settings
        )


    async def execute_task(self, task: str, user_request: str = None) -> TaskResult:
        # Use original user request for physics extraction if available
        extraction_source = user_request if user_request else task
        prompt = f'User Request: "{user_request}"\nStep: "{task}"\n\nCreate physics behavior based on the user request. Be creative for complex behaviors.'
        
        # Extract physics behavior for logging
        behavior = self._extract_physics_behavior_from_request(extraction_source)
        logger.debug(f"Extracted physics behavior: {behavior}")
        
        try:
            result = await self.agent.run(prompt)
            task_result = result.output
            logger.info(f"⚗️ PhysicsAgent: {task_result.explanation}")
            return task_result
        except Exception as e:
            logger.error(f"PhysicsAgent failed: {e}")
            
            # Try to extract raw response from pydantic-ai exception
            raw_response = None
            if hasattr(e, '_raw_response'):
                raw_response = e._raw_response
            elif hasattr(e, 'response'):
                raw_response = e.response
            elif hasattr(e, 'args') and e.args:
                # Sometimes the response is in the args
                for arg in e.args:
                    if isinstance(arg, str) and len(arg) > 10:
                        raw_response = arg
                        break
            
            if raw_response:
                logger.error(f"Raw LLM response: {raw_response}")
            else:
                logger.error(f"No raw response available. Exception type: {type(e)}")
                logger.error(f"Exception details: {str(e)}")
                logger.error(f"Exception dir: {[attr for attr in dir(e) if not attr.startswith('_')]}")
            
            raise Exception(f"PhysicsAgent failed to generate valid JSON. Raw error: {e}")
    
    def _extract_physics_behavior_from_request(self, request: str) -> str:
        request_lower = request.lower()
        
        if "bounc" in request_lower or "collision" in request_lower:
            return "bouncing"
        elif "repel" in request_lower or "repulse" in request_lower or "push away" in request_lower or "repulsion" in request_lower:
            return "repulsion"
        elif "flock" in request_lower or "birds" in request_lower or "cohesion" in request_lower:
            return "flocking"
        else:
            return "none"

class CompositionAgent(BaseAgent):
    def __init__(self):
        super().__init__("qwen2.5:3b")
        
        self.system_prompt_text = composition_agent_prompt.COMPOSITION_AGENT_PROMPT

        self.agent = Agent(
            model=self.model,
            output_type=GeneratedScript,
            system_prompt=self.system_prompt_text,
            model_settings=self.model_settings
        )

    async def compose_script(self, user_request: str, tool_calls: List[ToolCall]) -> GeneratedScript:
        try:
            logger.info(f"CompositionAgent: Generating script from user request and {len(tool_calls)} tool calls")
            
            # Build comprehensive prompt with user request and agent outputs
            prompt = self._build_composition_prompt(user_request, tool_calls)
            
            # Use LLM to generate the complete script
            result = await self.agent.run(prompt)
            generated_script = result.output
            
            logger.info(f"CompositionAgent: Generated script '{generated_script.title}' ({len(generated_script.code)} chars)")
            return generated_script
            
        except Exception as e:
            logger.error(f"CompositionAgent failed: {e}")
            
            # Try to extract raw response from pydantic-ai exception
            raw_response = None
            if hasattr(e, '_raw_response'):
                raw_response = e._raw_response
            elif hasattr(e, 'response'):
                raw_response = e.response
            elif hasattr(e, 'args') and e.args:
                # Sometimes the response is in the args
                for arg in e.args:
                    if isinstance(arg, str) and len(arg) > 10:
                        raw_response = arg
                        break
            
            if raw_response:
                logger.error(f"Raw LLM response: {raw_response}")
            else:
                logger.error(f"No raw response available. Exception type: {type(e)}")
                logger.error(f"Exception details: {str(e)}")
                logger.error(f"Exception dir: {[attr for attr in dir(e) if not attr.startswith('_')]}")
            
            raise Exception(f"CompositionAgent failed to generate valid JSON. Raw error: {e}")
    
    def _build_composition_prompt(self, user_request: str, tool_calls: List[ToolCall]) -> str:
        """Build comprehensive prompt for LLM script generation."""
        
        prompt = f"""Generate a complete Tölvera script for: "{user_request}"

AGENT OUTPUTS TO INTEGRATE:
"""
        
        # Add tool call information
        for i, tool_call in enumerate(tool_calls, 1):
            prompt += f"\n{i}. {tool_call.tool_name} - {tool_call.explanation}\n"
            if tool_call.parameters:
                prompt += f"   Parameters: {tool_call.parameters}\n"
            if tool_call.code_snippet:
                prompt += f"   Code: {tool_call.code_snippet}\n"
        
        # Extract key parameters for guidance
        particle_count = 3
        colors = []
        behaviors = []
        
        for tool_call in tool_calls:
            if "n" in tool_call.parameters:
                particle_count = tool_call.parameters["n"]
            if "color" in tool_call.parameters:
                colors.append(tool_call.parameters["color"])
            if "bouncing" in tool_call.explanation.lower():
                behaviors.append("bouncing")
            if "flocking" in tool_call.explanation.lower():
                behaviors.append("flocking")
        
        prompt += f"""

        KEY REQUIREMENTS:
        - Use {particle_count} particles
        - Implement the behaviors and properties specified above
        - Follow proper Tölvera patterns and structure
        - Generate complete, executable Python code
        - Include proper error handling in main execution

        Create a complete, working Tölvera script that fulfills the user's request."""
        
        return prompt
    

def create_agents():
    return {
        "conductor": ConductorAgent(),
        "particle": ParticleCreationAgent(),
        "color": ColorAgent(),
        "motion": MotionAgent(),
        "physics": PhysicsAgent(),
        "composition": CompositionAgent()
    }