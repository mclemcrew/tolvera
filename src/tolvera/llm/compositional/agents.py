# src/tolvera/llm/compositional/agents.py
"""
Expert agents for the MoE system.
Each agent specializes in a specific domain of Tölvera sketch generation.

Based on Table 2 from the architectural blueprint PDF.
"""

import logging
from typing import List, Dict, Any
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from .tools import (
    TaskResult, TaskPlan, GeneratedScript, ToolCall,
    PARTICLE_AGENT_TOOLS, COLOR_AGENT_TOOLS, MOTION_AGENT_TOOLS, PHYSICS_AGENT_TOOLS,
    generate_tolvera_imports, generate_main_function_template
)

logger = logging.getLogger(__name__)

# =============================================================================
# BASE AGENT CLASS
# =============================================================================

class BaseAgent:
    """Base class for all expert agents."""
    
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

# =============================================================================
# CONDUCTOR AGENT - Master Planner
# =============================================================================

class ConductorAgent(BaseAgent):
    """
    Master planner that decomposes user requests using Chain-of-Thought.
    Uses larger model for better reasoning capabilities.
    """
    
    def __init__(self):
        # Use larger model for planning
        super().__init__("llama3.2:3b")  # or "llama3.1:8b" if available
        
        system_prompt = """You are the master planner for Tölvera creative coding.
Your job is to analyze user requests and break them down into step-by-step plans that specialized expert agents can execute.

Available Expert Agents:
- ParticleCreationAgent: Creates and manages particles/pixels
- ColorPaletteAgent: Handles all color-related operations
- MotionDynamicsAgent: Manages movement, velocity, and forces
- PhysicsAgent: Implements complex physics interactions
- CompositionAgent: Assembles final Python script

THINK STEP BY STEP.
For "move a blue pixel from left to right":

1. ANALYSIS: User wants a single pixel (particle) that is blue and moves rightward
2. BREAKDOWN:
   - Need to create 1 particle → ParticleCreationAgent
   - Position it on left side → ParticleCreationAgent
   - Make it blue → ColorPaletteAgent
   - Give it rightward velocity → MotionDynamicsAgent
   - Assemble into complete script → CompositionAgent

3. PLAN: Create actionable steps for each expert

Always explain your reasoning clearly and create concrete, actionable plans.
Always end with a step for CompositionAgent to assemble the final script."""

        self.agent = Agent(
            model=self.model,
            result_type=TaskPlan,
            system_prompt=system_prompt
        )

    async def plan_task(self, user_request: str) -> TaskPlan:
        """Create detailed execution plan using Chain-of-Thought."""
        cot_prompt = f"""
Let's think step by step about this request: "{user_request}"

ANALYSIS:
- What objects/entities are needed?
- What properties must they have?
- What behaviors should they exhibit?
- What's the end goal?

EXPERT ROUTING:
- Which expert agents should handle which parts?
- What order should tasks be executed?

PLAN CREATION:
Create a step-by-step plan with specific tasks for each expert.
Always end with CompositionAgent to assemble the final script.

Request: {user_request}
"""
        
        try:
            result = await self.agent.run(cot_prompt)
            logger.info(f"📋 Conductor planned: {result.data.description}")
            return result.data
        except Exception as e:
            logger.error(f"❌ Planning failed: {e}")
            # Fallback plan
            return TaskPlan(
                description=f"Basic implementation of: {user_request}",
                steps=[
                    "Create required particles",
                    "Set colors as requested",
                    "Apply movement/forces",
                    "Assemble final script"
                ]
            )

# =============================================================================
# PARTICLE CREATION AGENT
# =============================================================================

class ParticleCreationAgent(BaseAgent):
    """Expert in creating and positioning particles."""
    
    def __init__(self):
        super().__init__("qwen2.5:3b")
        
        system_prompt = f"""You are a particle creation expert for Tölvera.
You handle creating particles and basic positioning.

{PARTICLE_AGENT_TOOLS}

For "create a pixel on the left": create_particles(1, 0, [0.1, 0.5])
For "create 5 particles": create_particles(5, 0, null)

Always output structured tool calls.
Focus only on creation and positioning."""

        self.agent = Agent(
            model=self.model,
            result_type=TaskResult,
            system_prompt=system_prompt
        )

    async def execute_task(self, task: str) -> TaskResult:
        """Execute particle creation task."""
        try:
            result = await self.agent.run(f"Task: {task}")
            logger.info(f"✨ ParticleCreationAgent: {result.data.explanation}")
            return result.data
        except Exception as e:
            logger.error(f"❌ Particle creation failed: {e}")
            return TaskResult(
                tool_calls=[
                    ToolCall(
                        tool_name="create_particles",
                        parameters={"n": 1, "species_id": 0, "position": [0.1, 0.5]}
                    )
                ],
                explanation="Created 1 particle as fallback"
            )

# =============================================================================
# COLOR PALETTE AGENT
# =============================================================================

class ColorPaletteAgent(BaseAgent):
    """Expert in color management."""
    
    def __init__(self):
        super().__init__("qwen2.5:3b")
        
        system_prompt = f"""You are a color expert for Tölvera.
You handle all color operations.

{COLOR_AGENT_TOOLS}

For "make it blue": set_species_color(0, [0.0, 0.0, 1.0, 1.0])

Focus only on color operations.
Use set_species_color for efficiency when coloring all particles of a species."""

        self.agent = Agent(
            model=self.model,
            result_type=TaskResult,
            system_prompt=system_prompt
        )

    async def execute_task(self, task: str, context: Dict[str, Any] = None) -> TaskResult:
        """Execute color task."""
        prompt = f"Task: {task}"
        if context:
            prompt += f"\nContext: {context}"
        
        try:
            result = await self.agent.run(prompt)
            logger.info(f"🎨 ColorPaletteAgent: {result.data.explanation}")
            return result.data
        except Exception as e:
            logger.error(f"❌ Color task failed: {e}")
            return TaskResult(
                tool_calls=[
                    ToolCall(
                        tool_name="set_species_color",
                        parameters={
                            "species_id": 0,
                            "color": [0.0, 0.0, 1.0, 1.0]  # Blue
                        }
                    )
                ],
                explanation="Applied blue color as fallback"
            )

# =============================================================================
# MOTION DYNAMICS AGENT
# =============================================================================

class MotionDynamicsAgent(BaseAgent):
    """Expert in movement and velocity."""
    
    def __init__(self):
        super().__init__("qwen2.5:3b")
        
        system_prompt = f"""You are a motion dynamics expert for Tölvera.
You handle movement and velocity.

{MOTION_AGENT_TOOLS}

For "move right": set_species_velocity(0, [2.0, 0.0])
For "varying speeds": apply_varying_speeds(0, 2.0, 1.0)

Focus on creating appropriate movement patterns.
Use set_species_velocity for efficiency when affecting all particles of a species."""

        self.agent = Agent(
            model=self.model,
            result_type=TaskResult,
            system_prompt=system_prompt
        )

    async def execute_task(self, task: str, context: Dict[str, Any] = None) -> TaskResult:
        """Execute motion task."""
        prompt = f"Task: {task}"
        if context:
            prompt += f"\nContext: {context}"
        
        try:
            result = await self.agent.run(prompt)
            logger.info(f"🚀 MotionDynamicsAgent: {result.data.explanation}")
            return result.data
        except Exception as e:
            logger.error(f"❌ Motion task failed: {e}")
            return TaskResult(
                tool_calls=[
                    ToolCall(
                        tool_name="set_species_velocity",
                        parameters={
                            "species_id": 0,
                            "velocity": [2.0, 0.0]  # Default rightward
                        }
                    )
                ],
                explanation="Applied rightward movement as fallback"
            )

# =============================================================================
# PHYSICS AGENT
# =============================================================================

class PhysicsAgent(BaseAgent):
    """Expert in complex physics interactions."""
    
    def __init__(self):
        super().__init__("qwen2.5:3b")
        
        system_prompt = f"""You are a physics expert for Tölvera.
You handle complex behaviors and interactions.

{PHYSICS_AGENT_TOOLS}

{PHYSICS_AGENT_TOOLS}

Focus on multi-particle behaviors, forces, and emergent systems.
For simple single-particle movement, defer to MotionDynamicsAgent."""

        self.agent = Agent(
            model=self.model,
            result_type=TaskResult,
            system_prompt=system_prompt
        )

    async def execute_task(self, task: str, context: Dict[str, Any] = None) -> TaskResult:
        """Execute physics task."""
        prompt = f"Task: {task}"
        if context:
            prompt += f"\nContext: {context}"
        
        try:
            result = await self.agent.run(prompt)
            logger.info(f"⚗️ PhysicsAgent: {result.data.explanation}")
            return result.data
        except Exception as e:
            logger.error(f"❌ Physics task failed: {e}")
            return TaskResult(
                tool_calls=[],
                explanation="No complex physics needed for this task"
            )

# =============================================================================
# COMPOSITION AGENT - Assembles Final Python Script
# =============================================================================

class CompositionAgent(BaseAgent):
    """
    Expert in assembling final Tölvera Python scripts.
    This converts tool calls into complete, runnable Python code.
    """
    
    def __init__(self):
        # Use larger model for code generation
        super().__init__("llama3.2:3b")
        
        system_prompt = """You are a code composition expert for Tölvera.
Your job is to take structured tool calls from other agents and assemble them into a complete, runnable Python script.

You will receive a list of tool calls like:
- create_particles(n=1, species_id=0, position=[0.1, 0.5])
- set_species_color(species_id=0, color=[0.0, 0.0, 1.0, 1.0])
- set_species_velocity(species_id=0, velocity=[2.0, 0.0])

Your task is to convert these into a complete Tölvera script with:
1. Proper imports
2. A main() function that sets up Tölvera
3. Initialization code that implements the tool calls
4. A @tv.render function that handles the animation loop
5. Proper error handling and cleanup

Template structure:
```python
\"\"\"
Brief description of what this script does.
\"\"\"

import taichi as ti
from tolvera import Tolvera, run

def main(**kwargs):
    \"\"\"Main function description.\"\"\"
    tv = Tolvera(n=NUM_PARTICLES, species=NUM_SPECIES, **kwargs)
    
    @ti.kernel
    def init_simulation():
        # Implement create_particles, set colors, etc.
        pass
    
    @ti.kernel  
    def update_simulation():
        # Implement movement, physics, etc.
        pass
    
    init_simulation()
    
    @tv.render
    def _():
        tv.px.background(0.0, 0.0, 0.0)  # Black background
        update_simulation()
        tv.px.particles(tv.p, tv.s.species(), "circle")
        return tv.px

if __name__ == '__main__':
    try:
        run(main)
    except KeyboardInterrupt:
        print("\\nExiting.")"""