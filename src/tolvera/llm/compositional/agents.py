# src/tolvera/llm/compositional/agents.py
"""
Robust expert agents with better JSON handling and fallbacks.
COMPLETELY FIXED VERSION with proper number extraction and physics behavior detection.
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
# BASE AGENT CLASS WITH ROBUST JSON HANDLING
# =============================================================================

class RobustBaseAgent:
    """Base class with comprehensive debugging of model responses."""
    
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
        """Safely run an agent with comprehensive debugging."""
        
        print(f"\n🔍 DEBUG: {self.__class__.__name__}")
        print(f"📝 Prompt: {prompt}")
        print(f"🎯 Expected type: {result_type}")
        
        # Step 1: Get raw model response first to see what it's actually generating
        try:
            system_prompt = self._get_agent_system_prompt(agent)
            
            raw_agent = Agent(
                model=self.model,
                result_type=str,
                system_prompt=system_prompt
            )
            
            raw_result = await raw_agent.run(prompt)
            raw_response = raw_result.output
            
            print(f"📥 Raw Model Response:")
            print(f"   Type: {type(raw_response)}")
            print(f"   Length: {len(raw_response)} chars")
            print(f"   Content: {repr(raw_response)}")
            
            # Step 2: Try to parse as JSON to see structure
            try:
                parsed_json = json.loads(raw_response)
                print(f"✅ JSON Parse: SUCCESS")
                print(f"   Keys: {list(parsed_json.keys())}")
                print(f"   Full JSON: {json.dumps(parsed_json, indent=2)}")
                
                # Step 3: Check against our schema validation
                if result_type == TaskPlan:
                    is_valid = validate_task_plan_json(parsed_json)
                    print(f"📋 TaskPlan validation: {is_valid}")
                    if is_valid:
                        try:
                            task_plan = TaskPlan(**parsed_json)
                            print(f"✅ TaskPlan creation: SUCCESS")
                            return task_plan
                        except Exception as e:
                            print(f"❌ TaskPlan creation failed: {e}")
                
                elif result_type == TaskResult:
                    is_valid = validate_tool_call_json(parsed_json)
                    print(f"✨ TaskResult validation: {is_valid}")
                    if is_valid:
                        try:
                            tool_calls = [ToolCall(**tc) for tc in parsed_json["tool_calls"]]
                            task_result = TaskResult(
                                tool_calls=tool_calls,
                                explanation=parsed_json["explanation"]
                            )
                            print(f"✅ TaskResult creation: SUCCESS")
                            return task_result
                        except Exception as e:
                            print(f"❌ TaskResult creation failed: {e}")
                
            except json.JSONDecodeError as e:
                print(f"❌ JSON Parse: FAILED - {e}")
                print(f"   Attempting JSON cleanup...")
                cleaned = self._clean_json_response(raw_response)
                print(f"   Cleaned: {repr(cleaned)}")
        
        except Exception as e:
            print(f"❌ Raw response failed: {e}")
        
        # Step 4: Try the original pydantic-ai agent
        print(f"\n🤖 Now trying pydantic-ai agent...")
        try:
            result = await agent.run(prompt)
            print(f"✅ Pydantic-AI: SUCCESS")
            print(f"   Result: {result.output}")
            return result.output
            
        except Exception as e:
            print(f"❌ Pydantic-AI: FAILED - {e}")
            print(f"   Error type: {type(e)}")
            
            if "Exceeded maximum retries" in str(e):
                print(f"   📊 Retry limit exceeded - this suggests validation issues")
        
        # Ultimate fallback
        print(f"🔄 Using fallback data")
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

    def _clean_json_response(self, response: str) -> str:
        """Clean JSON response for debugging."""
        response = response.strip()
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        
        start_idx = response.find('{')
        if start_idx == -1:
            return "{}"
        
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
        
        return response[start_idx:end_idx+1]

# =============================================================================
# ROBUST CONDUCTOR AGENT - ENHANCED
# =============================================================================

class RobustConductorAgent(RobustBaseAgent):
    """FIXED: Better task decomposition with number and behavior extraction."""
    
    def __init__(self):
        super().__init__("llama3.2:3b")
        
        self.system_prompt_text = """You are a task planner for Tölvera creative coding.

CRITICAL: Extract numbers and behaviors from requests accurately.

Examples of proper extraction:
- "three green particles bouncing around" → 3 particles + green color + bouncing physics
- "five blue pixels moving right" → 5 particles + blue color + rightward motion  
- "red circle moving in a circle" → 1 particle + red color + circular motion

You must respond with ONLY this exact JSON format:
{
    "description": "Brief description of the task",
    "steps": ["Step 1", "Step 2", "Step 3", "Step 4"]
}

STEP GENERATION RULES:
1. Always extract particle COUNT from numbers (one, two, three, 1, 2, 3, etc.)
2. Always extract COLORS (red, green, blue, yellow, etc.)
3. Always extract BEHAVIORS (bouncing, moving, rotating, floating, etc.)
4. Always include "Assemble final script" as last step

For "three green particles bouncing around":
{
    "description": "Create three green particles that bounce around the screen",
    "steps": [
        "Create 3 particles positioned randomly on screen",
        "Set particle color to green",
        "Apply bouncing physics with random velocities and collision detection", 
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
        """Create detailed execution plan with comprehensive debugging."""
        
        prompt = f'Create a 4-step plan for: "{user_request}"'
        
        fallback_plan = TaskPlan(
            description=f"Enhanced implementation of: {user_request}",
            steps=[
                f"Create particles based on request: {user_request}",
                "Set colors as specified in request",
                "Apply movement and physics behaviors", 
                "Assemble final script"
            ]
        )
        
        result = await self._safe_agent_run(self.agent, prompt, TaskPlan, fallback_plan)
        
        logger.info(f"📋 Conductor planned: {result.description}")
        return result

# =============================================================================
# ROBUST PARTICLE CREATION AGENT - ENHANCED
# =============================================================================

class RobustParticleCreationAgent(RobustBaseAgent):
    """FIXED: Better number extraction and positioning."""
    
    def __init__(self):
        super().__init__("qwen2.5:3b")
        
        self.system_prompt_text = """You are a particle creation expert for Tölvera.

CRITICAL: Extract exact particle counts from requests.

Number extraction examples:
- "3 particles" → n: 3
- "three particles" → n: 3
- "five blue pixels" → n: 5  
- "a red circle" → n: 1
- "particles" (no number) → n: 5 (default)

You must respond with ONLY this exact JSON format:
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
Use exact numbers from the request - don't default to 1 unless specifically "a particle" or "one particle".

CRITICAL: Respond with ONLY the JSON object."""

        self.agent = Agent(
            model=self.model,
            result_type=TaskResult,
            system_prompt=self.system_prompt_text
        )

    def _get_agent_system_prompt(self, agent: Agent) -> str:
        return self.system_prompt_text

    async def execute_task(self, task: str) -> TaskResult:
        """Execute particle creation with comprehensive debugging."""
        
        prompt = f'Task: "{task}"'
        
        fallback_result = TaskResult(
            tool_calls=[
                ToolCall(
                    tool_name="create_particles",
                    parameters={"n": 3, "species_id": 0, "position": None}
                )
            ],
            explanation="Created 3 particles as fallback"
        )
        
        result = await self._safe_agent_run(self.agent, prompt, TaskResult, fallback_result)
        
        logger.info(f"✨ ParticleCreationAgent: {result.explanation}")
        return result

# =============================================================================
# ROBUST COLOR PALETTE AGENT - UNCHANGED
# =============================================================================

class RobustColorPaletteAgent(RobustBaseAgent):
    """Robust color management expert with comprehensive debugging."""
    
    def __init__(self):
        super().__init__("qwen2.5:3b")
        
        self.system_prompt_text = """You are a color expert for Tölvera particles.

You must respond with ONLY this exact JSON format:
{
    "tool_calls": [
        {
            "tool_name": "set_species_color",
            "parameters": {"species_id": 0, "color": [R, G, B, 1.0]}
        }
    ],
    "explanation": "Brief explanation"
}

Colors (R, G, B, A from 0.0 to 1.0):
- red: [1.0, 0.0, 0.0, 1.0]
- green: [0.0, 1.0, 0.0, 1.0]
- blue: [0.0, 0.0, 1.0, 1.0]
- yellow: [1.0, 1.0, 0.0, 1.0]

CRITICAL: Respond with ONLY the JSON object."""

        self.agent = Agent(
            model=self.model,
            result_type=TaskResult,
            system_prompt=self.system_prompt_text
        )

    def _get_agent_system_prompt(self, agent: Agent) -> str:
        return self.system_prompt_text

    async def execute_task(self, task: str, context: Dict[str, Any] = None) -> TaskResult:
        """Execute color task with comprehensive debugging."""
        
        prompt = f'Task: "{task}"'
        
        fallback_result = TaskResult(
            tool_calls=[
                ToolCall(
                    tool_name="set_species_color",
                    parameters={"species_id": 0, "color": [0.0, 1.0, 0.0, 1.0]}
                )
            ],
            explanation="Applied green color as fallback"
        )
        
        result = await self._safe_agent_run(self.agent, prompt, TaskResult, fallback_result)
        
        logger.info(f"🎨 ColorPaletteAgent: {result.explanation}")
        return result

# =============================================================================
# ROBUST MOTION DYNAMICS AGENT - UNCHANGED
# =============================================================================

class RobustMotionDynamicsAgent(RobustBaseAgent):
    """Robust movement and velocity expert with comprehensive debugging."""
    
    def __init__(self):
        super().__init__("qwen2.5:3b")
        
        self.system_prompt_text = """You are a movement expert for Tölvera particles.

You must respond with ONLY this exact JSON format:
{
    "tool_calls": [
        {
            "tool_name": "set_species_velocity",
            "parameters": {"species_id": 0, "velocity": [VX, VY]}
        }
    ],
    "explanation": "Brief explanation"
}

Movement directions:
- Right: [2.0, 0.0]
- Left: [-2.0, 0.0] 
- Up: [0.0, -2.0]
- Down: [0.0, 2.0]

CRITICAL: Respond with ONLY the JSON object."""

        self.agent = Agent(
            model=self.model,
            result_type=TaskResult,
            system_prompt=self.system_prompt_text
        )

    def _get_agent_system_prompt(self, agent: Agent) -> str:
        return self.system_prompt_text

    async def execute_task(self, task: str, context: Dict[str, Any] = None) -> TaskResult:
        """Execute motion task with comprehensive debugging."""
        
        prompt = f'Task: "{task}"'
        
        fallback_result = TaskResult(
            tool_calls=[
                ToolCall(
                    tool_name="set_species_velocity",
                    parameters={"species_id": 0, "velocity": [2.0, 0.0]}
                )
            ],
            explanation="Applied rightward movement as fallback"
        )
        
        result = await self._safe_agent_run(self.agent, prompt, TaskResult, fallback_result)
        
        logger.info(f"🚀 MotionDynamicsAgent: {result.explanation}")
        return result

# =============================================================================
# ROBUST PHYSICS AGENT - ENHANCED
# =============================================================================

class RobustPhysicsAgent(RobustBaseAgent):
    """FIXED: Detect and implement bouncing and movement behaviors."""
    
    def __init__(self):
        super().__init__("qwen2.5:3b")
        
        self.system_prompt_text = """You are a physics expert for Tölvera particles.

BEHAVIOR DETECTION:
- "bouncing around" → apply_bouncing_behavior
- "moving randomly" → apply_random_movement  
- "floating" → apply_gentle_movement
- "spinning" → apply_rotational_movement

You must respond with ONLY this exact JSON format:

For bouncing/random movement:
{
    "tool_calls": [
        {
            "tool_name": "apply_bouncing_behavior",
            "parameters": {"species_id": 0, "speed_range": [1.0, 3.0], "collision_mode": "bounce"}
        }
    ],
    "explanation": "Applied bouncing physics with collision detection"
}

For gentle movement:
{
    "tool_calls": [
        {
            "tool_name": "apply_gentle_movement", 
            "parameters": {"species_id": 0, "speed": 1.0}
        }
    ],
    "explanation": "Applied gentle floating movement"
}

For no complex physics:
{
    "tool_calls": [],
    "explanation": "No complex physics needed for this task"
}

CRITICAL: Respond with ONLY the JSON object."""

        self.agent = Agent(
            model=self.model,
            result_type=TaskResult,
            system_prompt=self.system_prompt_text
        )

    def _get_agent_system_prompt(self, agent: Agent) -> str:
        return self.system_prompt_text

    async def execute_task(self, task: str, context: Dict[str, Any] = None) -> TaskResult:
        """Execute physics task with robust error handling."""
        
        prompt = f'Task: "{task}"'
        
        # Enhanced fallback that detects physics behaviors
        if "bounc" in task.lower() or "around" in task.lower():
            fallback_result = TaskResult(
                tool_calls=[
                    ToolCall(
                        tool_name="apply_bouncing_behavior",
                        parameters={"species_id": 0, "speed_range": [1.0, 3.0], "collision_mode": "bounce"}
                    )
                ],
                explanation="Applied bouncing physics with collision detection"
            )
        else:
            fallback_result = TaskResult(
                tool_calls=[],
                explanation="No complex physics needed for this task"
            )
        
        result = await self._safe_agent_run(self.agent, prompt, TaskResult, fallback_result)
        
        logger.info(f"⚗️ PhysicsAgent: {result.explanation}")
        return result

# =============================================================================
# COMPOSITION AGENT - ENHANCED
# =============================================================================

class RobustCompositionAgent(RobustBaseAgent):
    """Enhanced script composition with physics behavior support."""
    
    def __init__(self):
        super().__init__("llama3.2:3b")

    async def compose_script(self, user_request: str, tool_calls: List[ToolCall]) -> GeneratedScript:
        """Compose script using enhanced generation with physics behaviors."""
        
        try:
            # Use the enhanced script generation
            script_code = tool_calls_to_python_code_enhanced(tool_calls, user_request)
            
            logger.info(f"📝 CompositionAgent: Generated {len(script_code)} character script")
            
            return GeneratedScript(
                title=f"Generated: {user_request}",
                code=script_code,
                explanation=f"Enhanced implementation with {len(tool_calls)} tool calls: {user_request}"
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

# =============================================================================
# ENHANCED SCRIPT GENERATION
# =============================================================================

def tool_calls_to_python_code_enhanced(tool_calls: List[ToolCall], user_request: str) -> str:
    """
    ENHANCED: Generate scripts with proper physics behaviors and particle counts.
    """
    
    # Analyze tool calls
    particles_created = 1
    num_species = 1
    species_colors = {}
    physics_behaviors = {}
    positions = {}
    
    # Detect behaviors from request
    is_circular_motion = "circle" in user_request.lower() and "moving" in user_request.lower()
    has_bouncing = "bounc" in user_request.lower()
    has_random_movement = "around" in user_request.lower() or "random" in user_request.lower()
    
    for call in tool_calls:
        if call.tool_name == "create_particles":
            n = call.parameters.get("n", 1)
            particles_created = n
            positions[0] = call.parameters.get("position")
        
        elif call.tool_name == "set_species_color":
            species_id = call.parameters.get("species_id", 0)
            color = call.parameters.get("color", [1.0, 1.0, 1.0, 1.0])
            species_colors[species_id] = color
        
        elif call.tool_name in ["apply_bouncing_behavior", "apply_random_movement", "apply_gentle_movement"]:
            physics_behaviors[call.tool_name] = call.parameters
    
    # Generate different code based on behavior
    if is_circular_motion:
        return generate_circular_motion_script(user_request, particles_created, species_colors)
    elif has_bouncing or has_random_movement or physics_behaviors:
        return generate_bouncing_script(user_request, particles_created, species_colors, physics_behaviors)
    else:
        return generate_basic_movement_script(user_request, particles_created, species_colors, positions)

def generate_bouncing_script(user_request: str, particles_created: int, species_colors: dict, physics_behaviors: dict) -> str:
    """Generate script with bouncing/random movement physics."""
    
    color = species_colors.get(0, [0.0, 1.0, 0.0, 1.0])  # Default green
    
    return f'''"""
{user_request}
Generated by Tölvera MoE system.
"""

import taichi as ti
from tolvera import Tolvera, run

def main(**kwargs):
    """
    {user_request}
    """
    tv = Tolvera(n={particles_created}, species=1, **kwargs)

    @ti.kernel
    def init_particles():
        """Initialize particles with random positions and velocities."""
        # Deactivate all particles first
        for i in range(tv.pn):
            tv.p.field[i].active = 0.0

        # Initialize {particles_created} particles
        for i in range({particles_created}):
            # Random positions across the screen
            tv.p.field[i].pos = ti.Vector([
                ti.random() * tv.x,
                ti.random() * tv.y
            ])
            # Random velocities for bouncing
            speed = 1.0 + ti.random() * 2.0  # Speed between 1-3
            angle = ti.random() * 2.0 * 3.14159  # Random direction
            tv.p.field[i].vel = ti.Vector([
                speed * ti.cos(angle),
                speed * ti.sin(angle)
            ])
            tv.p.field[i].active = 1.0
            tv.p.field[i].species = 0
            tv.p.field[i].size = 12.0 + ti.random() * 8.0  # Size variation
            tv.p.field[i].mass = 1.0

    @ti.kernel
    def update_particles():
        """Update particle positions with bouncing physics."""
        for i in range({particles_created}):
            if tv.p.field[i].active > 0:
                # Update position
                tv.p.field[i].pos += tv.p.field[i].vel
                
                # Bounce off boundaries
                if tv.p.field[i].pos[0] <= 0 or tv.p.field[i].pos[0] >= tv.x:
                    tv.p.field[i].vel[0] *= -1.0
                    tv.p.field[i].pos[0] = ti.max(0.0, ti.min(tv.x, tv.p.field[i].pos[0]))
                
                if tv.p.field[i].pos[1] <= 0 or tv.p.field[i].pos[1] >= tv.y:
                    tv.p.field[i].vel[1] *= -1.0
                    tv.p.field[i].pos[1] = ti.max(0.0, ti.min(tv.y, tv.p.field[i].pos[1]))

    @ti.kernel
    def draw_particles():
        """Draw the bouncing particles."""
        tv.px.background(0.0, 0.0, 0.0)
        
        for i in range({particles_created}):
            if tv.p.field[i].active > 0:
                pos = tv.p.field[i].pos
                x = ti.cast(pos[0], ti.i32)
                y = ti.cast(pos[1], ti.i32)
                size = ti.cast(tv.p.field[i].size, ti.i32)
                
                species_id = tv.p.field[i].species
                color = tv.s.species.field[species_id].rgba
                
                tv.px.circle(x, y, size, color, fill=1)

    # Set colors
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
        print("\\nExiting.")
'''

def generate_circular_motion_script(user_request: str, particles_created: int, species_colors: dict) -> str:
    """Generate script with circular motion."""
    color = species_colors.get(0, [1.0, 0.0, 0.0, 1.0])  # Default red
    
    return f'''"""
{user_request}
Generated by Tölvera MoE system.
"""

import taichi as ti
from tolvera import Tolvera, run

def main(**kwargs):
    tv = Tolvera(n={particles_created}, species=1, **kwargs)

    center = ti.Vector.field(2, dtype=ti.f32, shape=())
    radius = ti.field(dtype=ti.f32, shape=())
    angle = ti.field(dtype=ti.f32, shape=())
    speed = ti.field(dtype=ti.f32, shape=())

    @ti.kernel
    def init_particles():
        center[None] = ti.Vector([tv.x * 0.5, tv.y * 0.5])
        radius[None] = min(tv.x, tv.y) * 0.2
        angle[None] = 0.0
        speed[None] = 0.03

        for i in range(tv.pn):
            tv.p.field[i].active = 0.0

        for i in range({particles_created}):
            tv.p.field[i].pos = center[None] + radius[None] * ti.Vector([ti.cos(angle[None]), ti.sin(angle[None])])
            tv.p.field[i].vel = ti.Vector([0.0, 0.0])
            tv.p.field[i].active = 1.0
            tv.p.field[i].species = 0
            tv.p.field[i].size = 16.0
            tv.p.field[i].mass = 1.0

    @ti.kernel
    def update_particles():
        angle[None] += speed[None]
        
        for i in range({particles_created}):
            if tv.p.field[i].active > 0:
                new_pos_x = center[None][0] + radius[None] * ti.cos(angle[None])
                new_pos_y = center[None][1] + radius[None] * ti.sin(angle[None])
                tv.p.field[i].pos = ti.Vector([new_pos_x, new_pos_y])

    @ti.kernel
    def draw_particles():
        tv.px.background(0.0, 0.0, 0.0)
        
        for i in range({particles_created}):
            if tv.p.field[i].active > 0:
                pos = tv.p.field[i].pos
                x = ti.cast(pos[0], ti.i32)
                y = ti.cast(pos[1], ti.i32)
                size = ti.cast(tv.p.field[i].size, ti.i32)
                
                species_id = tv.p.field[i].species
                color = tv.s.species.field[species_id].rgba
                
                tv.px.circle(x, y, size, color, fill=1)

    tv.s.species.field[0].rgba = ti.Vector({color})
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
        print("\\nExiting.")
'''

def generate_basic_movement_script(user_request: str, particles_created: int, species_colors: dict, positions: dict) -> str:
    """Generate script with basic linear movement."""
    color = species_colors.get(0, [0.0, 0.0, 1.0, 1.0])  # Default blue
    position = positions.get(0, [0.1, 0.5])  # Default left side
    
    return f'''"""
{user_request}
Generated by Tölvera MoE system.
"""

import taichi as ti
from tolvera import Tolvera, run

def main(**kwargs):
    tv = Tolvera(n={particles_created}, species=1, **kwargs)

    @ti.kernel
    def init_particles():
        for i in range(tv.pn):
            tv.p.field[i].active = 0.0

        for i in range({particles_created}):
            tv.p.field[i].pos = ti.Vector([tv.x * {position[0] if position else 0.1}, tv.y * {position[1] if position else 0.5}])
            tv.p.field[i].vel = ti.Vector([2.0, 0.0])
            tv.p.field[i].active = 1.0
            tv.p.field[i].species = 0
            tv.p.field[i].size = 16.0
            tv.p.field[i].mass = 1.0

    @ti.kernel
    def update_particles():
        for i in range({particles_created}):
            if tv.p.field[i].active > 0:
                tv.p.field[i].pos += tv.p.field[i].vel
                if tv.p.field[i].pos[0] > tv.x:
                    tv.p.field[i].pos[0] = 0.0
                if tv.p.field[i].pos[0] < 0:
                    tv.p.field[i].pos[0] = tv.x

    @ti.kernel
    def draw_particles():
        tv.px.background(0.0, 0.0, 0.0)
        
        for i in range({particles_created}):
            if tv.p.field[i].active > 0:
                pos = tv.p.field[i].pos
                x = ti.cast(pos[0], ti.i32)
                y = ti.cast(pos[1], ti.i32)
                size = ti.cast(tv.p.field[i].size, ti.i32)
                
                species_id = tv.p.field[i].species
                color = tv.s.species.field[species_id].rgba
                
                tv.px.circle(x, y, size, color, fill=1)

    tv.s.species.field[0].rgba = ti.Vector({color})
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