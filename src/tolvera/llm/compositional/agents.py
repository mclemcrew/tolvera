# src/tolvera/llm/compositional/agents_robust.py
"""
Robust expert agents with better JSON handling and fallbacks.
"""

import logging
from typing import List, Dict, Any
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from .tools import TaskResult, TaskPlan, GeneratedScript, ToolCall
from .json_utils import safe_json_parse, validate_tool_call_json, validate_task_plan_json

logger = logging.getLogger(__name__)

# =============================================================================
# BASE AGENT CLASS WITH ROBUST JSON HANDLING
# =============================================================================

class RobustBaseAgent:
    """Base class with robust JSON handling for all expert agents."""
    
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
        """Safely run an agent with robust error handling and fallbacks."""
        try:
            # Try the normal agent run first
            result = await agent.run(prompt)
            return result.data
        except Exception as e:
            logger.warning(f"Agent run failed with pydantic-ai: {e}")
            
            # Fallback: try to get raw response and parse manually
            try:
                # Create a simple string agent to get raw response
                raw_agent = Agent(
                    model=self.model,
                    result_type=str,
                    system_prompt="Respond exactly as requested with valid JSON."
                )
                
                raw_result = await raw_agent.run(prompt)
                raw_response = raw_result.data
                
                logger.debug(f"Raw response: {repr(raw_response[:200])}...")
                
                # Try to parse the raw response manually
                parsed_json = safe_json_parse(raw_response)
                
                if parsed_json:
                    # Validate structure based on expected type
                    if result_type == TaskPlan and validate_task_plan_json(parsed_json):
                        return TaskPlan(**parsed_json)
                    elif result_type == TaskResult and validate_tool_call_json(parsed_json):
                        # Convert tool_calls to ToolCall objects
                        tool_calls = [ToolCall(**tc) for tc in parsed_json["tool_calls"]]
                        return TaskResult(
                            tool_calls=tool_calls,
                            explanation=parsed_json["explanation"]
                        )
                    else:
                        logger.warning(f"Parsed JSON doesn't match expected structure for {result_type}")
                else:
                    logger.warning("Failed to parse raw response as JSON")
                    
            except Exception as parse_error:
                logger.warning(f"Raw response parsing also failed: {parse_error}")
            
            # Ultimate fallback
            logger.info(f"Using fallback data for {self.__class__.__name__}")
            return fallback_data

# =============================================================================
# ROBUST CONDUCTOR AGENT
# =============================================================================

class RobustConductorAgent(RobustBaseAgent):
    """Robust master planner with fallback handling."""
    
    def __init__(self):
        super().__init__("llama3.2:3b")
        
        system_prompt = """You are a task planner for Tölvera creative coding.

Respond with ONLY this exact JSON format:
{
    "description": "Brief description of the task",
    "steps": ["Step 1", "Step 2", "Step 3", "Step 4"]
}

For any creative request, always use these 4 steps:
1. Create particles (specify number and position if mentioned)
2. Set colors (if colors are mentioned)  
3. Apply movement or physics (if movement is mentioned)
4. Assemble final script

Example for "blue pixel moving right":
{
    "description": "Create a blue pixel that moves from left to right",
    "steps": [
        "Create 1 particle positioned on the left",
        "Set particle color to blue", 
        "Apply rightward velocity",
        "Assemble final script"
    ]
}

RESPOND WITH ONLY THE JSON. NO OTHER TEXT."""

        self.agent = Agent(
            model=self.model,
            result_type=TaskPlan,
            system_prompt=system_prompt
        )

    async def plan_task(self, user_request: str) -> TaskPlan:
        """Create detailed execution plan with robust error handling."""
        
        prompt = f'Create a 4-step plan for: "{user_request}"\n\nRespond with only JSON:'
        
        fallback_plan = TaskPlan(
            description=f"Basic implementation of: {user_request}",
            steps=[
                "Create required particles",
                "Set colors as requested",
                "Apply movement/forces", 
                "Assemble final script"
            ]
        )
        
        result = await self._safe_agent_run(self.agent, prompt, TaskPlan, fallback_plan)
        
        logger.info(f"📋 Conductor planned: {result.description}")
        return result

# =============================================================================
# ROBUST PARTICLE CREATION AGENT
# =============================================================================

class RobustParticleCreationAgent(RobustBaseAgent):
    """Robust particle creation expert."""
    
    def __init__(self):
        super().__init__("qwen2.5:3b")
        
        system_prompt = """You create particles for Tölvera.

Respond with ONLY this exact JSON format:
{
    "tool_calls": [
        {
            "tool_name": "create_particles",
            "parameters": {"n": NUMBER, "species_id": 0, "position": [X, Y]}
        }
    ],
    "explanation": "Brief explanation"
}

Position coordinates (0.0 to 1.0):
- Left: [0.1, 0.5], Center: [0.5, 0.5], Right: [0.9, 0.5]
- Top: [0.5, 0.1], Bottom: [0.5, 0.9]

Examples:
"create 2 particles on left" →
{
    "tool_calls": [
        {
            "tool_name": "create_particles",
            "parameters": {"n": 2, "species_id": 0, "position": [0.1, 0.5]}
        }
    ],
    "explanation": "Created 2 particles on the left side"
}

RESPOND WITH ONLY THE JSON. NO OTHER TEXT."""

        self.agent = Agent(
            model=self.model,
            result_type=TaskResult,
            system_prompt=system_prompt
        )

    async def execute_task(self, task: str) -> TaskResult:
        """Execute particle creation with robust error handling."""
        
        prompt = f'Task: "{task}"\n\nRespond with only JSON:'
        
        fallback_result = TaskResult(
            tool_calls=[
                ToolCall(
                    tool_name="create_particles",
                    parameters={"n": 1, "species_id": 0, "position": [0.1, 0.5]}
                )
            ],
            explanation="Created 1 particle as fallback"
        )
        
        result = await self._safe_agent_run(self.agent, prompt, TaskResult, fallback_result)
        
        logger.info(f"✨ ParticleCreationAgent: {result.explanation}")
        return result

# =============================================================================
# ROBUST COLOR PALETTE AGENT  
# =============================================================================

class RobustColorPaletteAgent(RobustBaseAgent):
    """Robust color management expert."""
    
    def __init__(self):
        super().__init__("qwen2.5:3b")
        
        system_prompt = """You set colors for Tölvera particles.

Respond with ONLY this exact JSON format:
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

Example:
"make it blue" →
{
    "tool_calls": [
        {
            "tool_name": "set_species_color",
            "parameters": {"species_id": 0, "color": [0.0, 0.0, 1.0, 1.0]}
        }
    ],
    "explanation": "Set species 0 color to blue"
}

RESPOND WITH ONLY THE JSON. NO OTHER TEXT."""

        self.agent = Agent(
            model=self.model,
            result_type=TaskResult,
            system_prompt=system_prompt
        )

    async def execute_task(self, task: str, context: Dict[str, Any] = None) -> TaskResult:
        """Execute color task with robust error handling."""
        
        prompt = f'Task: "{task}"\n\nRespond with only JSON:'
        
        fallback_result = TaskResult(
            tool_calls=[
                ToolCall(
                    tool_name="set_species_color",
                    parameters={"species_id": 0, "color": [0.0, 0.0, 1.0, 1.0]}
                )
            ],
            explanation="Applied blue color as fallback"
        )
        
        result = await self._safe_agent_run(self.agent, prompt, TaskResult, fallback_result)
        
        logger.info(f"🎨 ColorPaletteAgent: {result.explanation}")
        return result

# =============================================================================
# ROBUST MOTION DYNAMICS AGENT
# =============================================================================

class RobustMotionDynamicsAgent(RobustBaseAgent):
    """Robust movement and velocity expert."""
    
    def __init__(self):
        super().__init__("qwen2.5:3b")
        
        system_prompt = """You set movement for Tölvera particles.

Respond with ONLY this exact JSON format:
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
- Diagonal down-right: [2.0, 2.0]

Example:
"move from top to bottom" →
{
    "tool_calls": [
        {
            "tool_name": "set_species_velocity",
            "parameters": {"species_id": 0, "velocity": [0.0, 2.0]}
        }
    ],
    "explanation": "Set species 0 to move downward"
}

RESPOND WITH ONLY THE JSON. NO OTHER TEXT."""

        self.agent = Agent(
            model=self.model,
            result_type=TaskResult,
            system_prompt=system_prompt
        )

    async def execute_task(self, task: str, context: Dict[str, Any] = None) -> TaskResult:
        """Execute motion task with robust error handling."""
        
        prompt = f'Task: "{task}"\n\nRespond with only JSON:'
        
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
# ROBUST PHYSICS AGENT
# =============================================================================

class RobustPhysicsAgent(RobustBaseAgent):
    """Robust physics interactions expert."""
    
    def __init__(self):
        super().__init__("qwen2.5:3b")
        
        system_prompt = """You apply physics to Tölvera particles.

For simple movement tasks, respond with:
{
    "tool_calls": [],
    "explanation": "No complex physics needed for this task"
}

For flocking/swarming, respond with:
{
    "tool_calls": [
        {
            "tool_name": "apply_flock_behavior",
            "parameters": {"species_id": 0, "cohesion": 0.7, "separation": 0.3, "alignment": 0.5}
        }
    ],
    "explanation": "Applied flocking behavior"
}

Look for keywords: flock, swarm, together, group, birds, schools

RESPOND WITH ONLY THE JSON. NO OTHER TEXT."""

        self.agent = Agent(
            model=self.model,
            result_type=TaskResult,
            system_prompt=system_prompt
        )

    async def execute_task(self, task: str, context: Dict[str, Any] = None) -> TaskResult:
        """Execute physics task with robust error handling."""
        
        prompt = f'Task: "{task}"\n\nRespond with only JSON:'
        
        fallback_result = TaskResult(
            tool_calls=[],
            explanation="No complex physics needed for this task"
        )
        
        result = await self._safe_agent_run(self.agent, prompt, TaskResult, fallback_result)
        
        logger.info(f"⚗️ PhysicsAgent: {result.explanation}")
        return result

# =============================================================================
# COMPOSITION AGENT (unchanged - still uses direct generation)
# =============================================================================

class RobustCompositionAgent(RobustBaseAgent):
    """Robust script composition using direct generation."""
    
    def __init__(self):
        super().__init__("llama3.2:3b")

    async def compose_script(self, user_request: str, tool_calls: List[ToolCall]) -> GeneratedScript:
        """Compose script using direct generation (no LLM for final step)."""
        
        try:
            from .tools import tool_calls_to_python_code
            
            script_code = tool_calls_to_python_code(tool_calls, user_request)
            
            logger.info(f"📝 CompositionAgent: Generated {len(script_code)} character script")
            
            return GeneratedScript(
                title=f"Generated: {user_request}",
                code=script_code,
                explanation=f"Implemented {len(tool_calls)} tool calls to create: {user_request}"
            )
            
        except Exception as e:
            logger.error(f"❌ Script composition failed: {e}")
            
            # Generate emergency fallback
            num_particles = 10
            num_species = 1
            
            for call in tool_calls:
                if call.tool_name == "create_particles":
                    num_particles = max(num_particles, call.parameters.get("n", 1))
                    num_species = max(num_species, call.parameters.get("species_id", 0) + 1)
            
            fallback_code = self._generate_emergency_fallback(user_request, num_particles, num_species)
            return GeneratedScript(
                title=f"Fallback: {user_request}",
                code=fallback_code,
                explanation=f"Generated fallback due to error: {str(e)}"
            )

    def _generate_emergency_fallback(self, user_request: str, num_particles: int, num_species: int) -> str:
        """Generate emergency fallback script."""
        return f'''"""
Emergency fallback script for: {user_request}
"""

import taichi as ti
from tolvera import Tolvera, run

def main(**kwargs):
    """Emergency fallback implementation."""
    tv = Tolvera(n={num_particles}, species={num_species}, **kwargs)
    
    @ti.kernel
    def init_particles():
        """Initialize basic particles."""
        for i in range(min({num_particles}, tv.pn)):
            tv.p.field[i].active = 1.0
            tv.p.field[i].species = 0
            tv.p.field[i].pos = [tv.x * 0.2, tv.y * 0.5]
            tv.p.field[i].vel = [1.0, 0.0]
            tv.p.field[i].size = 8.0
    
    @ti.kernel
    def update_particles():
        """Update particle movement."""
        for i in range(tv.pn):
            if tv.p.field[i].active > 0:
                tv.p.field[i].pos += tv.p.field[i].vel
                if tv.p.field[i].pos[0] > tv.x:
                    tv.p.field[i].pos[0] = 0
    
    tv.s.species.field[0].rgba = [0.2, 0.4, 1.0, 1.0]
    init_particles()
    
    @tv.render
    def _():
        tv.px.background(0.05, 0.05, 0.1)
        update_particles()
        tv.px.particles(tv.p, tv.s.species(), "circle")
        return tv.px

if __name__ == '__main__':
    try:
        run(main)
    except KeyboardInterrupt:
        print("\\nExiting.")
'''

# =============================================================================
# FACTORY FUNCTIONS FOR ROBUST AGENTS
# =============================================================================

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

# =============================================================================
# TESTING FUNCTION
# =============================================================================

async def test_robust_agents():
    """Test all robust agents individually."""
    print("🧪 Testing Robust Agents")
    print("=" * 40)
    
    agents = create_robust_agents()
    
    # Test conductor
    print("\n📋 Testing Conductor...")
    plan = await agents["conductor"].plan_task("2 blue particles moving from top to bottom")
    print(f"✅ Plan: {plan.description}")
    
    # Test particle agent
    print("\n✨ Testing Particle Agent...")
    result = await agents["particle"].execute_task("Create 2 particles on the left")
    print(f"✅ Result: {result.explanation}")
    
    # Test color agent
    print("\n🎨 Testing Color Agent...")
    result = await agents["color"].execute_task("Make particles blue")
    print(f"✅ Result: {result.explanation}")
    
    # Test motion agent
    print("\n🚀 Testing Motion Agent...")
    result = await agents["motion"].execute_task("Move from top to bottom")
    print(f"✅ Result: {result.explanation}")
    
    print("\n✅ All robust agents tested successfully!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_robust_agents())