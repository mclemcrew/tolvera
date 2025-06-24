# src/tolvera/llm/compositional/agents.py
"""
IMPROVED robust expert agents with better JSON handling and more organic behaviors.
"""

import logging
from typing import List, Dict, Any
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import json

from .tools import TaskResult, TaskPlan, GeneratedScript, ToolCall
from .json_utils import safe_json_parse, validate_tool_call_json, validate_task_plan_json

def validate_taichi_code(code_snippet: str) -> bool:
    """Validate Taichi code for common syntax errors."""
    if not code_snippet or not code_snippet.strip():
        return True  # Empty is okay
    
    # Check for invalid ti.Vector usage (common error)
    if "ti.Vector(ti.random()" in code_snippet:
        logger.warning("❌ Invalid ti.Vector(ti.random()) - missing array brackets")
        return False
    
    if "ti.Vector(ti.random() *" in code_snippet:
        logger.warning("❌ Invalid ti.Vector(scalar) - needs [x, y] array")
        return False
    
    # Check for proper ti.Vector usage
    import re
    vector_pattern = r'ti\.Vector\([^[\]]*[^[\]]\)'  # ti.Vector(non-array)
    if re.search(vector_pattern, code_snippet) and "[" not in code_snippet:
        logger.warning("❌ ti.Vector should use array syntax: ti.Vector([x, y])")
        return False
    
    return True

logger = logging.getLogger(__name__)

# =============================================================================
# BASE AGENT CLASS WITH IMPROVED JSON HANDLING
# =============================================================================

class RobustBaseAgent:
    """Base class with improved model response handling."""
    
    def __init__(self, model_name: str = "qwen2.5:3b"):
        self.model_name = model_name
        self.model = OpenAIModel(
            model_name=model_name,
            provider=OpenAIProvider(
                base_url="http://localhost:11434/v1",
                api_key="ollama"
            )
        )
        logger.debug(f"Initialized {self.__class__.__name__} with model {model_name}")

    async def _safe_agent_run(self, agent: Agent, prompt: str, result_type: type, fallback_data: Any):
        """Safely run an agent with improved error handling."""
        
        try:
            # Run the agent (now returns string)
            result = await agent.run(prompt)
            raw_response = result.output
            logger.debug(f"✅ {self.__class__.__name__}: Got raw response")
            
            # Clean and parse manually
            parsed_json = safe_json_parse(raw_response)
            
            if parsed_json is not None:
                # Try to create the expected object
                if result_type == TaskPlan:
                    if validate_task_plan_json(parsed_json):
                        return TaskPlan(**parsed_json)
                    else:
                        logger.warning(f"TaskPlan validation failed: {parsed_json}")
                elif result_type == TaskResult:
                    if validate_tool_call_json(parsed_json):
                        tool_calls = [ToolCall(**tc) for tc in parsed_json["tool_calls"]]
                        
                        # Validate code snippets for syntax errors
                        for tool_call in tool_calls:
                            if hasattr(tool_call, 'code_snippet') and tool_call.code_snippet:
                                if not validate_taichi_code(tool_call.code_snippet):
                                    logger.warning(f"❌ Invalid Taichi code detected, using fallback")
                                    return fallback_data
                        
                        return TaskResult(
                            tool_calls=tool_calls,
                            explanation=parsed_json["explanation"]
                        )
                    else:
                        logger.warning(f"TaskResult validation failed: {parsed_json}")
            else:
                logger.warning(f"Failed to parse JSON from: {raw_response[:200]}...")
                
        except Exception as e:
            logger.warning(f"⚠️ {self.__class__.__name__}: Agent execution failed: {e}")
        
        # Ultimate fallback
        logger.warning(f"🔄 {self.__class__.__name__}: Using fallback data")
        return fallback_data

    def _get_agent_system_prompt(self, agent: Agent) -> str:
        """Get the system prompt from an agent safely."""
        try:
            if hasattr(agent, 'system_prompt'):
                if callable(agent.system_prompt):
                    return agent.system_prompt()
                else:
                    return agent.system_prompt
            return "You are a helpful assistant. Respond with valid JSON."
        except Exception:
            return "You are a helpful assistant. Respond with valid JSON."

# =============================================================================
# IMPROVED CONDUCTOR AGENT
# =============================================================================

class RobustConductorAgent(RobustBaseAgent):
    """Improved conductor with better task decomposition."""
    
    def __init__(self):
        super().__init__("llama3.2:3b")
        
        self.system_prompt_text = """You are a task planner for Tölvera creative coding.

Your job is to break down natural language requests into clear, actionable steps.

IMPORTANT: You must respond with ONLY a JSON object in this exact format:
{
    "description": "Brief description of the overall goal",
    "steps": ["Step 1", "Step 2", "Step 3", "Step 4"]
}

Rules for steps:
1. Always include "Create particles" as the first step (specify the number if mentioned)
2. Always include "Set colors" as the second step (if colors are mentioned)
3. Include movement/physics steps as needed
4. Always end with "Assemble final script"

Examples:

For "three blue particles bouncing around":
{
    "description": "Create three blue particles that bounce around the screen",
    "steps": [
        "Create 3 particles positioned randomly on screen",
        "Set particle color to blue", 
        "Apply bouncing physics with collision detection",
        "Assemble final script"
    ]
}

For "red circle moving in a circle":
{
    "description": "Create a red circle that moves in circular motion",
    "steps": [
        "Create 1 particle positioned at center",
        "Set particle color to red",
        "Apply circular motion physics",
        "Assemble final script"
    ]
}

CRITICAL: Respond with ONLY the JSON object. No other text."""

        self.agent = Agent(
            model=self.model,
            result_type=str,
            system_prompt=self.system_prompt_text
        )

    def _get_agent_system_prompt(self, agent: Agent) -> str:
        return self.system_prompt_text

    async def plan_task(self, user_request: str) -> TaskPlan:
        """Create detailed execution plan."""
        
        prompt = f'Create a 4-step plan for: "{user_request}"'
        
        fallback_plan = TaskPlan(
            description=f"Create particle system for: {user_request}",
            steps=[
                "Create particles based on request",
                "Set colors as specified", 
                "Apply movement and physics behaviors",
                "Assemble final script"
            ]
        )
        
        result = await self._safe_agent_run(self.agent, prompt, TaskPlan, fallback_plan)
        
        logger.info(f"📋 Conductor planned: {result.description}")
        return result

# =============================================================================
# IMPROVED PARTICLE CREATION AGENT
# =============================================================================

class RobustParticleCreationAgent(RobustBaseAgent):
    """Improved particle creation with better number extraction."""
    
    def __init__(self):
        super().__init__("qwen2.5:3b")
        
        self.system_prompt_text = """You are a particle creation expert for Tölvera.

You generate BOTH structured tool calls AND actual Tölvera code for creating particles.

Extract particle counts and properties from requests:
- "3 particles" → n: 3
- "different velocities" → individual velocity arrays like [2.0, 3.5, 1.2]
- "random positions" → scattered placement using ti.random()
- "staggered" → offset positions like y positions 100, 250, 400

IMPORTANT: You must respond with ONLY this JSON format:
{
    "tool_calls": [
        {
            "tool_name": "create_particles",
            "parameters": {"n": NUMBER, "species_id": 0, "position": null},
            "code_snippet": "for i in range(3):\n    tv.p.field[i].pos = ti.Vector([tv.x * ti.random(), tv.y * ti.random()])\n    tv.p.field[i].vel = ti.Vector([2.0 + i * 0.5, 0.0])\n    tv.p.field[i].active = 1.0\n    tv.p.field[i].species = 0\n    tv.p.field[i].size = 16.0\n    tv.p.field[i].mass = 1.0",
            "explanation": "What this code creates"
        }
    ],
    "explanation": "Overall explanation"
}

Your code_snippet should use EXACT Tölvera patterns:
- Position: tv.p.field[i].pos = ti.Vector([x_value, y_value])  # ALWAYS 2 values in list
- Velocity: tv.p.field[i].vel = ti.Vector([vx_value, vy_value])  # ALWAYS 2 values in list  
- Screen coords: tv.x * ti.random(), tv.y * ti.random() for random positions
- Fixed coords: 50.0 + i * 100.0 for spaced positions
- NEVER use: ti.Vector(single_value) - this is INVALID
- Set .active=1.0, .species=0, .size=16.0, .mass=1.0

CRITICAL: Follow EXACT syntax or code will crash!"""

        self.agent = Agent(
            model=self.model,
            result_type=str,
            system_prompt=self.system_prompt_text
        )

    def _get_agent_system_prompt(self, agent: Agent) -> str:
        return self.system_prompt_text

    async def execute_task(self, task: str) -> TaskResult:
        """Execute particle creation."""
        
        prompt = f'Task: "{task}"'
        
        # Extract number from task for better fallback
        n_particles = self._extract_number_from_task(task)
        
        # Generate fallback code based on extracted number with better velocity handling
        fallback_code = f"""
# Create {n_particles} particles with individual properties
for i in range({n_particles}):
    tv.p.field[i].pos = ti.Vector([tv.x * ti.random(), tv.y * ti.random()])
    tv.p.field[i].vel = ti.Vector([2.0 + i * 0.5, 0.0])  # Different velocities
    tv.p.field[i].active = 1.0
    tv.p.field[i].species = 0
    tv.p.field[i].size = 16.0
    tv.p.field[i].mass = 1.0"""
        
        fallback_result = TaskResult(
            tool_calls=[
                ToolCall(
                    tool_name="create_particles",
                    parameters={"n": n_particles, "species_id": 0, "position": None},
                    code_snippet=fallback_code,
                    explanation=f"Created {n_particles} particles with random positions"
                )
            ],
            explanation=f"Generated particle creation code for {n_particles} particles"
        )
        
        result = await self._safe_agent_run(self.agent, prompt, TaskResult, fallback_result)
        
        logger.info(f"✨ ParticleCreationAgent: {result.explanation}")
        return result
    
    def _extract_number_from_task(self, task: str) -> int:
        """Extract number from task string."""
        import re
        
        # Number word mapping
        number_words = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "a": 1, "single": 1
        }
        
        task_lower = task.lower()
        
        # Check for number words
        for word, num in number_words.items():
            if word in task_lower:
                return num
        
        # Check for digits
        numbers = re.findall(r'\b(\d+)\b', task)
        if numbers:
            return int(numbers[0])
        
        # Default
        return 3

# =============================================================================
# IMPROVED COLOR AGENT
# =============================================================================

class RobustColorPaletteAgent(RobustBaseAgent):
    """Improved color management with better color detection."""
    
    def __init__(self):
        super().__init__("qwen2.5:3b")
        
        self.system_prompt_text = """You are a color expert for Tölvera particles.

You generate BOTH structured tool calls AND actual Tölvera code for particle colors.

Color capabilities in Tölvera:
- Species colors: tv.s.species.field[0].rgba = ti.Vector([R, G, B, A])
- Individual particle colors: set colors per particle if needed
- Standard color values as RGBA arrays

Standard colors (RGBA values):
- red: [1.0, 0.0, 0.0, 1.0]
- green: [0.0, 1.0, 0.0, 1.0] 
- blue: [0.0, 0.0, 1.0, 1.0]
- yellow: [1.0, 1.0, 0.0, 1.0]
- white: [1.0, 1.0, 1.0, 1.0]
- orange: [1.0, 0.5, 0.0, 1.0]
- purple: [0.5, 0.0, 1.0, 1.0]

IMPORTANT: You must respond with ONLY this JSON format:
{
    "tool_calls": [
        {
            "tool_name": "set_species_color",
            "parameters": {"species_id": 0, "color": [R, G, B, A]},
            "code_snippet": "tv.s.species.field[0].rgba = ti.Vector([0.0, 1.0, 0.0, 1.0])",
            "explanation": "What colors were applied"
        }
    ],
    "explanation": "Overall color explanation"
}

Your code_snippet should use Tölvera patterns:
- Species color setting: tv.s.species.field[0].rgba = ti.Vector([r, g, b, a])
- Use exact RGBA values from the standard colors
- Create ti.Vector() for color values

CRITICAL: Generate real working Tölvera color code!"""

        self.agent = Agent(
            model=self.model,
            result_type=str,
            system_prompt=self.system_prompt_text
        )

    def _get_agent_system_prompt(self, agent: Agent) -> str:
        return self.system_prompt_text

    async def execute_task(self, task: str, context: Dict[str, Any] = None) -> TaskResult:
        """Execute color task."""
        
        prompt = f'Task: "{task}"'
        
        # Extract color from task for better fallback
        color = self._extract_color_from_task(task)
        color_name = self._get_color_name(color)
        
        # Generate fallback color code
        fallback_code = f"""
# Set species color to {color_name}
tv.s.species.field[0].rgba = ti.Vector({color})"""
        
        fallback_result = TaskResult(
            tool_calls=[
                ToolCall(
                    tool_name="set_species_color",
                    parameters={"species_id": 0, "color": color},
                    code_snippet=fallback_code,
                    explanation=f"Set species color to {color_name}"
                )
            ],
            explanation=f"Generated color code for {color_name}"
        )
        
        result = await self._safe_agent_run(self.agent, prompt, TaskResult, fallback_result)
        
        logger.info(f"🎨 ColorPaletteAgent: {result.explanation}")
        return result
    
    def _extract_color_from_task(self, task: str) -> List[float]:
        """Extract color from task string."""
        color_map = {
            "red": [1.0, 0.0, 0.0, 1.0],
            "green": [0.0, 1.0, 0.0, 1.0], 
            "blue": [0.0, 0.0, 1.0, 1.0],
            "yellow": [1.0, 1.0, 0.0, 1.0],
            "orange": [1.0, 0.5, 0.0, 1.0],
            "purple": [0.5, 0.0, 1.0, 1.0],
            "white": [1.0, 1.0, 1.0, 1.0],
        }
        
        task_lower = task.lower()
        for color_name, color_value in color_map.items():
            if color_name in task_lower:
                return color_value
        
        return [0.0, 1.0, 0.0, 1.0]  # Default green
    
    def _get_color_name(self, color: List[float]) -> str:
        """Get color name from RGBA values."""
        color_names = {
            (1.0, 0.0, 0.0, 1.0): "red",
            (0.0, 1.0, 0.0, 1.0): "green",
            (0.0, 0.0, 1.0, 1.0): "blue",
            (1.0, 1.0, 0.0, 1.0): "yellow",
        }
        return color_names.get(tuple(color), "unknown")

# =============================================================================
# IMPROVED MOTION AGENT
# =============================================================================

class RobustMotionDynamicsAgent(RobustBaseAgent):
    """Improved movement with better direction detection."""
    
    def __init__(self):
        super().__init__("qwen2.5:3b")
        
        self.system_prompt_text = """You are a movement expert for Tölvera particles.

You generate BOTH structured tool calls AND actual Tölvera code for particle movement.

Movement patterns you can create:
- Basic movement: tv.p.field[i].pos += tv.p.field[i].vel
- Individual particle movement: different behavior per particle
- Directional movement: right [2.0, 0.0], left [-2.0, 0.0], up [0.0, -2.0], down [0.0, 2.0]
- Mathematical patterns: sine waves, circular motion using ti.sin(), ti.cos()
- Boundary handling: wrapping around screen edges

IMPORTANT: You must respond with ONLY this JSON format:
{
    "tool_calls": [
        {
            "tool_name": "set_species_velocity",
            "parameters": {"species_id": 0, "velocity": [2.0, 0.0]},
            "code_snippet": "for i in range(3):\n    if tv.p.field[i].active > 0:\n        tv.p.field[i].pos += tv.p.field[i].vel\n        if tv.p.field[i].pos[0] > tv.x:\n            tv.p.field[i].pos[0] = 0.0",
            "explanation": "What this movement code does"
        }
    ],
    "explanation": "Overall movement explanation"
}

Your code_snippet should use Tölvera patterns:
- Particle access: tv.p.field[i].pos, tv.p.field[i].vel, tv.p.field[i].active
- Screen dimensions: tv.x, tv.y for boundaries
- Update positions: tv.p.field[i].pos += tv.p.field[i].vel
- Handle boundaries with wrapping or bouncing
- Use proper loop range based on particle count

CRITICAL: Generate real working Tölvera movement code!"""

        self.agent = Agent(
            model=self.model,
            result_type=str,
            system_prompt=self.system_prompt_text
        )

    def _get_agent_system_prompt(self, agent: Agent) -> str:
        return self.system_prompt_text

    async def execute_task(self, task: str, context: Dict[str, Any] = None) -> TaskResult:
        """Execute motion task."""
        
        prompt = f'Task: "{task}"'
        
        # Extract velocity from task for better fallback
        velocity = self._extract_velocity_from_task(task)
        
        # Generate fallback movement code
        fallback_code = f"""
# Apply movement to particles
for i in range(tv.pn):
    if tv.p.field[i].active > 0:
        tv.p.field[i].pos += tv.p.field[i].vel
        # Wrap around screen boundaries
        if tv.p.field[i].pos[0] > tv.x:
            tv.p.field[i].pos[0] = 0.0
        if tv.p.field[i].pos[0] < 0:
            tv.p.field[i].pos[0] = tv.x"""
        
        fallback_result = TaskResult(
            tool_calls=[
                ToolCall(
                    tool_name="set_species_velocity",
                    parameters={"species_id": 0, "velocity": velocity},
                    code_snippet=fallback_code,
                    explanation=f"Applied movement with velocity {velocity} and screen wrapping"
                )
            ],
            explanation=f"Generated movement code with velocity {velocity}"
        )
        
        result = await self._safe_agent_run(self.agent, prompt, TaskResult, fallback_result)
        
        logger.info(f"🚀 MotionDynamicsAgent: {result.explanation}")
        return result
    
    def _extract_velocity_from_task(self, task: str) -> List[float]:
        """Extract velocity from task string."""
        task_lower = task.lower()
        
        if "right" in task_lower or "left to right" in task_lower:
            return [2.0, 0.0]
        elif "left" in task_lower:
            return [-2.0, 0.0]
        elif "up" in task_lower or "upward" in task_lower:
            return [0.0, -2.0]
        elif "down" in task_lower or "downward" in task_lower or "top to bottom" in task_lower:
            return [0.0, 2.0]
        else:
            return [2.0, 0.0]  # Default right

# =============================================================================
# IMPROVED PHYSICS AGENT
# =============================================================================

class RobustPhysicsAgent(RobustBaseAgent):
    """Improved physics with better behavior detection."""
    
    def __init__(self):
        super().__init__("qwen2.5:3b")
        
        self.system_prompt_text = """You are a physics expert for Tölvera particles.

You generate BOTH structured tool calls AND actual Tölvera code for particle physics.

Physics behaviors you can create in Tölvera:
- Bouncing: particles reverse velocity when hitting boundaries
- Flocking: cohesion, separation, alignment like birds
- Boundary physics: wrapping, bouncing, or constraining to screen
- Simple forces and interactions

IMPORTANT: You must respond with ONLY this JSON format:

For bouncing behavior:
{
    "tool_calls": [
        {
            "tool_name": "apply_bouncing_behavior",
            "parameters": {"species_id": 0, "speed_range": [1.0, 3.0], "collision_mode": "bounce"},
            "code_snippet": "if tv.p.field[i].pos[0] <= 0 or tv.p.field[i].pos[0] >= tv.x:\n    tv.p.field[i].vel[0] *= -1.0\n    tv.p.field[i].pos[0] = ti.max(0.0, ti.min(tv.x, tv.p.field[i].pos[0]))",
            "explanation": "Applied bouncing physics with boundary collision"
        }
    ],
    "explanation": "Generated bouncing physics code"
}

For no physics:
{
    "tool_calls": [],
    "explanation": "No physics behavior needed for this request"
}

Your code_snippet should use Tölvera patterns:
- Particle access: tv.p.field[i].pos, tv.p.field[i].vel, tv.p.field[i].active
- Screen boundaries: tv.x, tv.y
- Physics calculations with ti.max(), ti.min(), ti.sin(), ti.cos()
- Boundary collision detection and response

CRITICAL: Generate real working Tölvera physics code!"""

        self.agent = Agent(
            model=self.model,
            result_type=str,
            system_prompt=self.system_prompt_text
        )

    def _get_agent_system_prompt(self, agent: Agent) -> str:
        return self.system_prompt_text

    async def execute_task(self, task: str, context: Dict[str, Any] = None) -> TaskResult:
        """Execute physics task."""
        
        prompt = f'Task: "{task}"'
        
        # Determine physics behavior from task
        physics_tool = self._determine_physics_from_task(task)
        
        result = await self._safe_agent_run(self.agent, prompt, TaskResult, physics_tool)
        
        logger.info(f"⚗️ PhysicsAgent: {result.explanation}")
        return result
    
    def _determine_physics_from_task(self, task: str) -> TaskResult:
        """Determine physics behavior from task and generate code."""
        task_lower = task.lower()
        
        if "bounc" in task_lower or "collision" in task_lower:
            bouncing_code = """
# Bouncing physics - reverse velocity at boundaries
for i in range(tv.pn):
    if tv.p.field[i].active > 0:
        # Bounce off left/right walls
        if tv.p.field[i].pos[0] <= 0 or tv.p.field[i].pos[0] >= tv.x:
            tv.p.field[i].vel[0] *= -1.0
            tv.p.field[i].pos[0] = ti.max(0.0, ti.min(tv.x, tv.p.field[i].pos[0]))
        
        # Bounce off top/bottom walls
        if tv.p.field[i].pos[1] <= 0 or tv.p.field[i].pos[1] >= tv.y:
            tv.p.field[i].vel[1] *= -1.0
            tv.p.field[i].pos[1] = ti.max(0.0, ti.min(tv.y, tv.p.field[i].pos[1]))"""
            
            return TaskResult(
                tool_calls=[
                    ToolCall(
                        tool_name="apply_physics_behavior",
                        parameters={"behavior_type": "bouncing", "strength": 1.0},
                        code_snippet=bouncing_code,
                        explanation="Applied bouncing physics with boundary collision"
                    )
                ],
                explanation="Generated bouncing physics code"
            )
        elif "flock" in task_lower or "birds" in task_lower or "cohesion" in task_lower:
            flocking_code = """
# Basic flocking behavior - particles attracted to each other
for i in range(tv.pn):
    if tv.p.field[i].active > 0:
        center = ti.Vector([0.0, 0.0])
        count = 0
        
        # Find center of nearby particles
        for j in range(tv.pn):
            if tv.p.field[j].active > 0 and i != j:
                dist = (tv.p.field[i].pos - tv.p.field[j].pos).norm()
                if dist < 100.0:  # Within flocking radius
                    center += tv.p.field[j].pos
                    count += 1
        
        # Move toward center
        if count > 0:
            center /= count
            direction = (center - tv.p.field[i].pos).normalized()
            tv.p.field[i].vel += direction * 0.1  # Cohesion strength"""
            
            return TaskResult(
                tool_calls=[
                    ToolCall(
                        tool_name="apply_physics_behavior",
                        parameters={"behavior_type": "flocking", "strength": 0.1},
                        code_snippet=flocking_code,
                        explanation="Applied flocking behavior like birds"
                    )
                ],
                explanation="Generated flocking physics code"
            )
        else:
            return TaskResult(
                tool_calls=[],
                explanation="No complex physics needed for this task"
            )

# =============================================================================
# IMPROVED COMPOSITION AGENT
# =============================================================================

# Fixed composition agent section for agents.py

class RobustCompositionAgent(RobustBaseAgent):
    """Code-weaving composition agent that combines code snippets into complete Tölvera scripts."""
    
    def __init__(self):
        super().__init__("llama3.2:3b")

    async def compose_script(self, user_request: str, tool_calls: List[ToolCall]) -> GeneratedScript:
        """Weave code snippets from tool calls into a complete Tölvera script."""
        
        try:
            logger.info(f"📝 CompositionAgent: Weaving {len(tool_calls)} code snippets into script")
            
            # Extract code snippets from tool calls
            particle_code = ""
            color_code = ""
            motion_code = ""
            physics_code = ""
            
            # Categorize code snippets
            for tool_call in tool_calls:
                if tool_call.code_snippet:
                    if "create_particles" in tool_call.tool_name:
                        particle_code += tool_call.code_snippet + "\n"
                    elif "apply_particle_colors" in tool_call.tool_name:
                        color_code += tool_call.code_snippet + "\n"
                    elif "apply_particle_movement" in tool_call.tool_name:
                        motion_code += tool_call.code_snippet + "\n"
                    elif "apply_physics" in tool_call.tool_name:
                        physics_code += tool_call.code_snippet + "\n"
            
            # Weave together the complete script
            script_code = self._weave_complete_script(
                user_request, particle_code, color_code, motion_code, physics_code, tool_calls
            )
            
            # Validate the generated script
            if len(script_code) < 200:  # Script too short
                logger.warning("Generated script seems too short, using enhanced fallback")
                script_code = self._generate_enhanced_fallback(user_request, tool_calls)
            
            logger.info(f"📝 CompositionAgent: Wove {len(script_code)} character Python script")
            
            return GeneratedScript(
                title=f"Generated: {user_request}",
                code=script_code,
                explanation=f"Woven script from {len(tool_calls)} code snippets: {user_request}"
            )
            
        except Exception as e:
            logger.error(f"❌ Script composition failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Generate enhanced emergency fallback
            fallback_code = self._generate_enhanced_fallback(user_request, tool_calls)
            return GeneratedScript(
                title=f"Fallback: {user_request}",
                code=fallback_code,
                explanation=f"Generated enhanced fallback due to error: {str(e)}"
            )
    
    def _weave_complete_script(self, user_request: str, particle_code: str, color_code: str, 
                              motion_code: str, physics_code: str, tool_calls: List[ToolCall]) -> str:
        """Weave individual code snippets into a complete Tölvera script."""
        
        # Extract particle count from tool calls
        particle_count = 3  # Default
        for tool_call in tool_calls:
            if "n" in tool_call.parameters:
                particle_count = tool_call.parameters["n"]
                break
        
        # Build the complete script
        script = f'"""\n{user_request}\nGenerated by Tölvera MoE system with code weaving.\n"""\n\n'
        script += "import taichi as ti\nfrom tolvera import Tolvera, run\n\n"
        script += f"def main(**kwargs):\n"
        script += f"    tv = Tolvera(n={particle_count}, species=1, **kwargs)\n\n"
        
        # Init particles kernel
        script += "    @ti.kernel\n"
        script += "    def init_particles():\n"
        script += "        # Deactivate all particles first\n"
        script += "        for i in range(tv.pn):\n"
        script += "            tv.p.field[i].active = 0.0\n\n"
        
        if particle_code.strip():
            # Add particle creation code with proper indentation
            script += "        # Particle creation\n"
            for line in particle_code.strip().split('\n'):
                if line.strip():
                    script += f"        {line}\n"
        else:
            # Default particle creation with improved velocity handling
            script += f"        # Default particle creation\n"
            script += f"        for i in range({particle_count}):\n"
            script += f"            tv.p.field[i].pos = ti.Vector([tv.x * ti.random(), tv.y * ti.random()])\n"
            
            # Check if user requested different velocities
            if "different velocities" in user_request.lower() or "different speeds" in user_request.lower():
                script += f"            tv.p.field[i].vel = ti.Vector([2.0 + i * 0.5, 0.0])  # Different velocities\n"
            else:
                script += f"            tv.p.field[i].vel = ti.Vector([2.0, 0.0])\n"
            
            script += f"            tv.p.field[i].active = 1.0\n"
            script += f"            tv.p.field[i].species = 0\n"
            script += f"            tv.p.field[i].size = 16.0\n"
            script += f"            tv.p.field[i].mass = 1.0\n"
        
        script += "\n"
        
        # Update particles kernel
        script += "    @ti.kernel\n"
        script += "    def update_particles():\n"
        
        if motion_code.strip():
            script += "        # Motion code\n"
            for line in motion_code.strip().split('\n'):
                if line.strip():
                    script += f"        {line}\n"
        
        if physics_code.strip():
            script += "        # Physics code\n"
            for line in physics_code.strip().split('\n'):
                if line.strip():
                    script += f"        {line}\n"
        
        if not motion_code.strip() and not physics_code.strip():
            # Default update
            script += f"        for i in range({particle_count}):\n"
            script += f"            if tv.p.field[i].active > 0:\n"
            script += f"                tv.p.field[i].pos += tv.p.field[i].vel\n"
            script += f"                # Wrap around screen\n"
            script += f"                if tv.p.field[i].pos[0] > tv.x:\n"
            script += f"                    tv.p.field[i].pos[0] = 0.0\n"
            script += f"                if tv.p.field[i].pos[0] < 0:\n"
            script += f"                    tv.p.field[i].pos[0] = tv.x\n"
        
        script += "\n"
        
        # Draw particles kernel
        script += "    @ti.kernel\n"
        script += "    def draw_particles():\n"
        script += "        tv.px.background(0.0, 0.0, 0.0)\n\n"
        script += f"        for i in range({particle_count}):\n"
        script += "            if tv.p.field[i].active > 0:\n"
        script += "                pos = tv.p.field[i].pos\n"
        script += "                x = ti.cast(pos[0], ti.i32)\n"
        script += "                y = ti.cast(pos[1], ti.i32)\n"
        script += "                size = ti.cast(tv.p.field[i].size, ti.i32)\n\n"
        script += "                species_id = tv.p.field[i].species\n"
        script += "                color = tv.s.species.field[species_id].rgba\n\n"
        script += "                tv.px.circle(x, y, size, color, fill=1)\n\n"
        
        # Colors setup
        if color_code.strip():
            script += "    # Color setup\n"
            for line in color_code.strip().split('\n'):
                if line.strip():
                    script += f"    {line}\n"
        else:
            script += "    # Default green color\n"
            script += "    tv.s.species.field[0].rgba = ti.Vector([0.0, 1.0, 0.0, 1.0])\n"
        
        script += "\n"
        
        # Initialization and render loop
        script += "    initialized = ti.field(ti.i32, shape=())\n"
        script += "    initialized[None] = 0\n\n"
        script += "    @tv.render\n"
        script += "    def _():\n"
        script += "        if initialized[None] == 0:\n"
        script += "            init_particles()\n"
        script += "            initialized[None] = 1\n\n"
        script += "        update_particles()\n"
        script += "        draw_particles()\n\n"
        script += "        return tv.px\n\n"
        
        # Main execution
        script += "if __name__ == '__main__':\n"
        script += "    try:\n"
        script += "        run(main)\n"
        script += "    except KeyboardInterrupt:\n"
        script += '        print("\\nExiting.")\n'
        
        return script

    def _generate_enhanced_fallback(self, user_request: str, tool_calls: List[ToolCall]) -> str:
        """Generate enhanced fallback script with better analysis."""
        
        # Extract information from tool calls
        particles_created = 1
        color = [0.0, 1.0, 0.0, 1.0]  # Default green
        velocity = [2.0, 0.0]  # Default rightward movement
        has_bouncing = False
        
        try:
            for call in tool_calls:
                if call.tool_name == "create_particles":
                    particles_created = call.parameters.get("n", 1)
                elif call.tool_name == "set_species_color":
                    color = call.parameters.get("color", [0.0, 1.0, 0.0, 1.0])
                elif call.tool_name == "set_species_velocity":
                    velocity = call.parameters.get("velocity", [2.0, 0.0])
                elif call.tool_name == "apply_bouncing_behavior":
                    has_bouncing = True
        except Exception as e:
            logger.warning(f"Error extracting tool call info: {e}")
        
        # Detect colors from request
        if "blue" in user_request.lower():
            color = [0.0, 0.0, 1.0, 1.0]
        elif "red" in user_request.lower():
            color = [1.0, 0.0, 0.0, 1.0]
        elif "yellow" in user_request.lower():
            color = [1.0, 1.0, 0.0, 1.0]
        
        # Detect movement from request
        if "right" in user_request.lower():
            velocity = [2.0, 0.0]
        elif "left" in user_request.lower():
            velocity = [-2.0, 0.0]
        elif "up" in user_request.lower():
            velocity = [0.0, -2.0]
        elif "down" in user_request.lower():
            velocity = [0.0, 2.0]
        
        # Detect bouncing
        if "bounc" in user_request.lower() or "around" in user_request.lower():
            has_bouncing = True
        
        return self._generate_enhanced_fallback_code(user_request, particles_created, color, velocity, has_bouncing)

    def _generate_enhanced_fallback_code(self, user_request: str, particles_created: int, 
                                       color: List[float], velocity: List[float], has_bouncing: bool) -> str:
        """Generate the actual enhanced fallback code."""
        
        return f'''"""
Enhanced fallback script for: {user_request}
Generated by Tölvera MoE system enhanced composition fallback.
"""

import taichi as ti
from tolvera import Tolvera, run

def main(**kwargs):
    """
    Enhanced fallback implementation for: {user_request}
    """
    tv = Tolvera(n={particles_created}, species=1, **kwargs)

    @ti.kernel
    def init_particles():
        # Deactivate all particles first
        for i in range(tv.pn):
            tv.p.field[i].active = 0.0

        # Initialize {particles_created} particles
        for i in range({particles_created}):
            tv.p.field[i].active = 1.0
            tv.p.field[i].species = 0
            tv.p.field[i].pos = ti.Vector([tv.x * 0.2, tv.y * 0.5])  # Start left side
            {"tv.p.field[i].vel = ti.Vector([2.0 + i * 0.5, 0.0])  # Different velocities" if "different velocities" in user_request.lower() or "different speeds" in user_request.lower() else f"tv.p.field[i].vel = ti.Vector({velocity})"}
            tv.p.field[i].size = 16.0
            tv.p.field[i].mass = 1.0

    @ti.kernel
    def update_particles():
        for i in range({particles_created}):
            if tv.p.field[i].active > 0:
                # Update position
                tv.p.field[i].pos += tv.p.field[i].vel
                
                {"# Bouncing physics" if has_bouncing else "# Wrapping physics"}
                if tv.p.field[i].pos[0] <= 0 or tv.p.field[i].pos[0] >= tv.x:
                    {"tv.p.field[i].vel[0] *= -1.0" if has_bouncing else "tv.p.field[i].pos[0] = tv.x if tv.p.field[i].pos[0] < 0 else 0.0"}
                if tv.p.field[i].pos[1] <= 0 or tv.p.field[i].pos[1] >= tv.y:
                    {"tv.p.field[i].vel[1] *= -1.0" if has_bouncing else "tv.p.field[i].pos[1] = tv.y if tv.p.field[i].pos[1] < 0 else 0.0"}

    @ti.kernel
    def draw_particles():
        tv.px.background(0.0, 0.0, 0.0)
        for i in range({particles_created}):
            if tv.p.field[i].active > 0:
                pos = tv.p.field[i].pos
                x = ti.cast(pos[0], ti.i32)
                y = ti.cast(pos[1], ti.i32)
                size = ti.cast(tv.p.field[i].size, ti.i32)
                tv.px.circle(x, y, size, ti.Vector({color}), fill=1)

    # Set species color
    tv.s.species.field[0].rgba = ti.Vector({color})
    
    # Initialization flag
    initialized = ti.field(ti.i32, shape=())
    initialized[None] = 0
    
    @tv.render
    def _():
        if initialized[None] == 0:
            init_particles()
            initialized[None] = 1
        
        update_particles()
        draw_particles()
        
        return tv.px

if __name__ == '__main__':
    try:
        run(main)
    except KeyboardInterrupt:
        print("\\nExiting enhanced fallback script.")
'''

def create_robust_agents():
    """Create all robust expert agents."""
    return {
        "conductor": RobustConductorAgent(),
        "particle": RobustParticleCreationAgent(),
        "color": RobustColorPaletteAgent(),
        "motion": RobustMotionDynamicsAgent(),
        "physics": RobustPhysicsAgent(),
        "composition": RobustCompositionAgent()
    }