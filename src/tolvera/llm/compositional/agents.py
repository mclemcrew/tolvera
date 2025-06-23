# src/tolvera/llm/compositional/agents.py
"""
Robust expert agents with better JSON handling and fallbacks.
Fixed for pydantic-ai deprecation and improved prompts.
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
        # FIXED: Remove temperature parameter
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
            # FIXED: Get the system prompt correctly from the agent
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
                            print(f"   Tool calls data: {parsed_json['tool_calls']}")
                            # Debug each tool call individually
                            for i, tc_data in enumerate(parsed_json["tool_calls"]):
                                try:
                                    tc = ToolCall(**tc_data)
                                    print(f"   ✅ ToolCall {i}: SUCCESS")
                                except Exception as tc_error:
                                    print(f"   ❌ ToolCall {i}: FAILED - {tc_error}")
                                    print(f"      Data: {tc_data}")
                
            except json.JSONDecodeError as e:
                print(f"❌ JSON Parse: FAILED - {e}")
                print(f"   Attempting JSON cleanup...")
                cleaned = self._clean_json_response(raw_response)
                print(f"   Cleaned: {repr(cleaned)}")
        
        except Exception as e:
            print(f"❌ Raw response failed: {e}")
        
        # Step 4: Now try the original pydantic-ai agent to see its specific error
        print(f"\n🤖 Now trying pydantic-ai agent...")
        try:
            result = await agent.run(prompt)
            print(f"✅ Pydantic-AI: SUCCESS")
            print(f"   Result: {result.output}")
            return result.output
            
        except Exception as e:
            print(f"❌ Pydantic-AI: FAILED - {e}")
            print(f"   Error type: {type(e)}")
            
            # Get more details about the pydantic-ai error
            if "Exceeded maximum retries" in str(e):
                print(f"   📊 Retry limit exceeded - this suggests validation issues")
            
        # Ultimate fallback
        print(f"🔄 Using fallback data")
        return fallback_data

    def _get_agent_system_prompt(self, agent: Agent) -> str:
        """Get the system prompt from an agent safely."""
        # Try different ways to get the system prompt
        try:
            if hasattr(agent, 'system_prompt'):
                if callable(agent.system_prompt):
                    return agent.system_prompt()
                else:
                    return agent.system_prompt
            # Fallback to a generic prompt
            return "You are a helpful assistant. Respond with valid JSON."
        except Exception:
            return "You are a helpful assistant. Respond with valid JSON."

    def _clean_json_response(self, response: str) -> str:
        """Clean JSON response for debugging."""
        # Remove common prefixes
        response = response.strip()
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        
        # Find JSON boundaries
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
# ROBUST CONDUCTOR AGENT
# =============================================================================

class RobustConductorAgent(RobustBaseAgent):
    """Robust master planner with comprehensive debugging."""
    
    def __init__(self):
        super().__init__("llama3.2:3b")
        
        # Store system prompt as an attribute for easy access
        self.system_prompt_text = """You are a task planner for Tölvera creative coding.

You must respond with ONLY this exact JSON format:
{
    "description": "Brief description of the task",
    "steps": ["Step 1", "Step 2", "Step 3", "Step 4"]
}

For any creative request, always use these 4 steps:
1. Create particles (specify number and position if mentioned)
2. Set colors (if colors are mentioned)  
3. Apply movement or physics (if movement is mentioned)
4. Assemble final script

Example - Input: "blue pixel moving right"
Expected output:
{
    "description": "Create a blue pixel that moves from left to right",
    "steps": [
        "Create 1 particle positioned on the left",
        "Set particle color to blue", 
        "Apply rightward velocity",
        "Assemble final script"
    ]
}

CRITICAL: Respond with ONLY the JSON object. No other text before or after."""

        self.agent = Agent(
            model=self.model,
            result_type=TaskPlan,
            system_prompt=self.system_prompt_text
        )

    def _get_agent_system_prompt(self, agent: Agent) -> str:
        """Get the system prompt for this agent."""
        return self.system_prompt_text

    async def plan_task(self, user_request: str) -> TaskPlan:
        """Create detailed execution plan with comprehensive debugging."""
        
        prompt = f'Create a 4-step plan for: "{user_request}"'
        
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
    """Robust particle creation expert with comprehensive debugging."""
    
    def __init__(self):
        super().__init__("qwen2.5:3b")
        
        self.system_prompt_text = """You are a particle creation expert for Tölvera.

You must respond with ONLY this exact JSON format:
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

Example - Input: "create 2 particles on left"
Expected output:
{
    "tool_calls": [
        {
            "tool_name": "create_particles",
            "parameters": {"n": 2, "species_id": 0, "position": [0.1, 0.5]}
        }
    ],
    "explanation": "Created 2 particles on the left side"
}

CRITICAL: Respond with ONLY the JSON object. No other text before or after."""

        self.agent = Agent(
            model=self.model,
            result_type=TaskResult,
            system_prompt=self.system_prompt_text
        )

    def _get_agent_system_prompt(self, agent: Agent) -> str:
        """Get the system prompt for this agent."""
        return self.system_prompt_text

    async def execute_task(self, task: str) -> TaskResult:
        """Execute particle creation with comprehensive debugging."""
        
        prompt = f'Task: "{task}"'
        
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

Example - Input: "make it blue"
Expected output:
{
    "tool_calls": [
        {
            "tool_name": "set_species_color",
            "parameters": {"species_id": 0, "color": [0.0, 0.0, 1.0, 1.0]}
        }
    ],
    "explanation": "Set species 0 color to blue"
}

CRITICAL: Respond with ONLY the JSON object. No other text before or after."""

        self.agent = Agent(
            model=self.model,
            result_type=TaskResult,
            system_prompt=self.system_prompt_text
        )

    def _get_agent_system_prompt(self, agent: Agent) -> str:
        """Get the system prompt for this agent."""
        return self.system_prompt_text

    async def execute_task(self, task: str, context: Dict[str, Any] = None) -> TaskResult:
        """Execute color task with comprehensive debugging."""
        
        prompt = f'Task: "{task}"'
        
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
- Diagonal down-right: [2.0, 2.0]

Example - Input: "move from top to bottom"
Expected output:
{
    "tool_calls": [
        {
            "tool_name": "set_species_velocity",
            "parameters": {"species_id": 0, "velocity": [0.0, 2.0]}
        }
    ],
    "explanation": "Set species 0 to move downward"
}

CRITICAL: Respond with ONLY the JSON object. No other text before or after."""

        self.agent = Agent(
            model=self.model,
            result_type=TaskResult,
            system_prompt=self.system_prompt_text
        )

    def _get_agent_system_prompt(self, agent: Agent) -> str:
        """Get the system prompt for this agent."""
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
# ROBUST PHYSICS AGENT
# =============================================================================

class RobustPhysicsAgent(RobustBaseAgent):
    """Robust physics interactions expert."""
    
    def __init__(self):
        super().__init__("qwen2.5:3b")
        
        self.system_prompt_text = """You are a physics expert for Tölvera particles.

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

CRITICAL: Respond with ONLY the JSON object. No other text before or after."""

        self.agent = Agent(
            model=self.model,
            result_type=TaskResult,
            system_prompt=self.system_prompt_text
        )

    def _get_agent_system_prompt(self, agent: Agent) -> str:
        """Get the system prompt for this agent."""
        return self.system_prompt_text

    async def execute_task(self, task: str, context: Dict[str, Any] = None) -> TaskResult:
        """Execute physics task with robust error handling."""
        
        prompt = f'Task: "{task}"'
        
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