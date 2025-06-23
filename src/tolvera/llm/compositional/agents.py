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
            # Try the pydantic-ai agent directly first
            result = await agent.run(prompt)
            logger.debug(f"✅ {self.__class__.__name__}: Direct success")
            return result.output
            
        except Exception as e:
            logger.warning(f"⚠️ {self.__class__.__name__}: Pydantic-AI failed: {e}")
            
            # Try manual approach with raw model
            try:
                system_prompt = self._get_agent_system_prompt(agent)
                
                raw_agent = Agent(
                    model=self.model,
                    result_type=str,
                    system_prompt=system_prompt
                )
                
                raw_result = await raw_agent.run(prompt)
                raw_response = raw_result.output
                
                # Clean and parse manually
                parsed_json = safe_json_parse(raw_response)
                
                if parsed_json is not None:
                    # Try to create the expected object
                    if result_type == TaskPlan:
                        if validate_task_plan_json(parsed_json):
                            return TaskPlan(**parsed_json)
                    elif result_type == TaskResult:
                        if validate_tool_call_json(parsed_json):
                            tool_calls = [ToolCall(**tc) for tc in parsed_json["tool_calls"]]
                            return TaskResult(
                                tool_calls=tool_calls,
                                explanation=parsed_json["explanation"]
                            )
                
            except Exception as manual_error:
                logger.error(f"❌ {self.__class__.__name__}: Manual parsing failed: {manual_error}")
        
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
            result_type=TaskPlan,
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

Extract particle counts accurately from requests:
- "3 particles" → n: 3
- "three particles" → n: 3  
- "a particle" → n: 1
- "particles" (no number) → n: 5 (default)

IMPORTANT: You must respond with ONLY this JSON format:
{
    "tool_calls": [
        {
            "tool_name": "create_particles",
            "parameters": {"n": NUMBER, "species_id": 0, "position": null}
        }
    ],
    "explanation": "Created N particles"
}

Position should be null for random placement unless specific position mentioned.

CRITICAL: 
- Only use "create_particles" tool name (not "create_particle")
- Use exact numbers from the request
- Use valid JSON (no Math.random() or JavaScript)
- Respond with ONLY the JSON object"""

        self.agent = Agent(
            model=self.model,
            result_type=TaskResult,
            system_prompt=self.system_prompt_text
        )

    def _get_agent_system_prompt(self, agent: Agent) -> str:
        return self.system_prompt_text

    async def execute_task(self, task: str) -> TaskResult:
        """Execute particle creation."""
        
        prompt = f'Task: "{task}"'
        
        # Extract number from task for better fallback
        n_particles = self._extract_number_from_task(task)
        
        fallback_result = TaskResult(
            tool_calls=[
                ToolCall(
                    tool_name="create_particles",
                    parameters={"n": n_particles, "species_id": 0, "position": None}
                )
            ],
            explanation=f"Created {n_particles} particles"
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

IMPORTANT: You must respond with ONLY this JSON format:
{
    "tool_calls": [
        {
            "tool_name": "set_species_color",
            "parameters": {"species_id": 0, "color": [R, G, B, 1.0]}
        }
    ],
    "explanation": "Set color to [color name]"
}

Standard colors (use exact RGBA values):
- red: [1.0, 0.0, 0.0, 1.0]
- green: [0.0, 1.0, 0.0, 1.0]
- blue: [0.0, 0.0, 1.0, 1.0]
- yellow: [1.0, 1.0, 0.0, 1.0]
- white: [1.0, 1.0, 1.0, 1.0]
- orange: [1.0, 0.5, 0.0, 1.0]
- purple: [0.5, 0.0, 1.0, 1.0]

CRITICAL: Respond with ONLY the JSON object."""

        self.agent = Agent(
            model=self.model,
            result_type=TaskResult,
            system_prompt=self.system_prompt_text
        )

    def _get_agent_system_prompt(self, agent: Agent) -> str:
        return self.system_prompt_text

    async def execute_task(self, task: str, context: Dict[str, Any] = None) -> TaskResult:
        """Execute color task."""
        
        prompt = f'Task: "{task}"'
        
        # Extract color from task for better fallback
        color = self._extract_color_from_task(task)
        
        fallback_result = TaskResult(
            tool_calls=[
                ToolCall(
                    tool_name="set_species_color",
                    parameters={"species_id": 0, "color": color}
                )
            ],
            explanation=f"Set color to {self._get_color_name(color)}"
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

IMPORTANT: You must respond with ONLY this JSON format:
{
    "tool_calls": [
        {
            "tool_name": "set_species_velocity",
            "parameters": {"species_id": 0, "velocity": [VX, VY]}
        }
    ],
    "explanation": "Applied [direction] movement"
}

Movement directions:
- Right: [2.0, 0.0]
- Left: [-2.0, 0.0] 
- Up: [0.0, -2.0]
- Down: [0.0, 2.0]
- Top to bottom: [0.0, 2.0]
- Left to right: [2.0, 0.0]

CRITICAL: Respond with ONLY the JSON object."""

        self.agent = Agent(
            model=self.model,
            result_type=TaskResult,
            system_prompt=self.system_prompt_text
        )

    def _get_agent_system_prompt(self, agent: Agent) -> str:
        return self.system_prompt_text

    async def execute_task(self, task: str, context: Dict[str, Any] = None) -> TaskResult:
        """Execute motion task."""
        
        prompt = f'Task: "{task}"'
        
        # Extract velocity from task for better fallback
        velocity = self._extract_velocity_from_task(task)
        
        fallback_result = TaskResult(
            tool_calls=[
                ToolCall(
                    tool_name="set_species_velocity",
                    parameters={"species_id": 0, "velocity": velocity}
                )
            ],
            explanation=f"Applied movement with velocity {velocity}"
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

IMPORTANT: You must respond with ONLY this JSON format:

For bouncing behavior:
{
    "tool_calls": [
        {
            "tool_name": "apply_bouncing_behavior",
            "parameters": {"species_id": 0, "speed_range": [1.0, 3.0], "collision_mode": "bounce"}
        }
    ],
    "explanation": "Applied bouncing physics with collision detection"
}

For flocking behavior:
{
    "tool_calls": [
        {
            "tool_name": "apply_flock_behavior", 
            "parameters": {"species_id": 0, "cohesion": 0.8, "separation": 0.6, "alignment": 0.7}
        }
    ],
    "explanation": "Applied flocking behavior like birds"
}

For no complex physics:
{
    "tool_calls": [],
    "explanation": "No complex physics needed for this task"
}

CRITICAL: 
- Only use tool names that exist: apply_bouncing_behavior, apply_flock_behavior
- Do NOT use: compute_cohesion_force, compute_separation_force, etc.
- Respond with ONLY the JSON object"""

        self.agent = Agent(
            model=self.model,
            result_type=TaskResult,
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
        """Determine physics behavior from task."""
        task_lower = task.lower()
        
        if "bounc" in task_lower or "collision" in task_lower:
            return TaskResult(
                tool_calls=[
                    ToolCall(
                        tool_name="apply_bouncing_behavior",
                        parameters={"species_id": 0, "speed_range": [1.0, 3.0], "collision_mode": "bounce"}
                    )
                ],
                explanation="Applied bouncing physics with collision detection"
            )
        elif "flock" in task_lower or "birds" in task_lower or "cohesion" in task_lower:
            return TaskResult(
                tool_calls=[
                    ToolCall(
                        tool_name="apply_flock_behavior",
                        parameters={"species_id": 0, "cohesion": 0.8, "separation": 0.6, "alignment": 0.7}
                    )
                ],
                explanation="Applied flocking behavior like birds"
            )
        else:
            return TaskResult(
                tool_calls=[],
                explanation="No complex physics needed for this task"
            )

# =============================================================================
# IMPROVED COMPOSITION AGENT
# =============================================================================

class RobustCompositionAgent(RobustBaseAgent):
    """Improved script composition."""
    
    def __init__(self):
        super().__init__("llama3.2:3b")

    async def compose_script(self, user_request: str, tool_calls: List[ToolCall]) -> GeneratedScript:
        """Compose script using enhanced generation."""
        
        try:
            # Import the enhanced script generation function
            from .tools import tool_calls_to_python_code
            
            script_code = tool_calls_to_python_code(tool_calls, user_request)
            
            logger.info(f"📝 CompositionAgent: Generated {len(script_code)} character script")
            
            return GeneratedScript(
                title=f"Generated: {user_request}",
                code=script_code,
                explanation=f"Implementation with {len(tool_calls)} tool calls: {user_request}"
            )
            
        except Exception as e:
            logger.error(f"❌ Script composition failed: {e}")
            
            # Generate emergency fallback
            fallback_code = self._generate_emergency_fallback(user_request)
            return GeneratedScript(
                title=f"Fallback: {user_request}",
                code=fallback_code,
                explanation=f"Generated fallback due to error: {str(e)}"
            )

    def _generate_emergency_fallback(self, user_request: str) -> str:
        """Generate emergency fallback script."""
        return f'''"""
Emergency fallback script for: {user_request}
"""

import taichi as ti
from tolvera import Tolvera, run

def main(**kwargs):
    tv = Tolvera(n=3, species=1, **kwargs)
    
    @ti.kernel
    def init_particles():
        for i in range(3):
            tv.p.field[i].active = 1.0
            tv.p.field[i].species = 0
            tv.p.field[i].pos = ti.Vector([tv.x * ti.random(), tv.y * ti.random()])
            tv.p.field[i].vel = ti.Vector([ti.random() * 4 - 2, ti.random() * 4 - 2])
            tv.p.field[i].size = 12.0
    
    @ti.kernel
    def update_particles():
        for i in range(3):
            if tv.p.field[i].active > 0:
                tv.p.field[i].pos += tv.p.field[i].vel
                if tv.p.field[i].pos[0] <= 0 or tv.p.field[i].pos[0] >= tv.x:
                    tv.p.field[i].vel[0] *= -1
                if tv.p.field[i].pos[1] <= 0 or tv.p.field[i].pos[1] >= tv.y:
                    tv.p.field[i].vel[1] *= -1
    
    @ti.kernel
    def draw_particles():
        tv.px.background(0.0, 0.0, 0.0)
        for i in range(3):
            if tv.p.field[i].active > 0:
                pos = tv.p.field[i].pos
                x = ti.cast(pos[0], ti.i32)
                y = ti.cast(pos[1], ti.i32)
                size = ti.cast(tv.p.field[i].size, ti.i32)
                tv.px.circle(x, y, size, ti.Vector([0.0, 1.0, 0.0, 1.0]), fill=1)
    
    tv.s.species.field[0].rgba = ti.Vector([0.0, 1.0, 0.0, 1.0])
    initialized = ti.field(ti.i32, shape=())
    
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
        print("\\nExiting.")
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