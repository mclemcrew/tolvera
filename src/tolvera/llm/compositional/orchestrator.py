# src/tolvera/llm/compositional/orchestrator.py
"""
Main orchestrator that coordinates the MoE system to generate complete Tölvera scripts.
This implements the full architectural blueprint from the PDF.

Pipeline:
1. ConductorAgent: Plans the task using Chain-of-Thought
2. Expert Agents: Generate structured tool calls for their domains
3. CompositionAgent: Assembles tool calls into a complete Python script
"""

import logging
from typing import List, Dict, Any
from .agents import (
    ConductorAgent, ParticleCreationAgent, ColorPaletteAgent,
    MotionDynamicsAgent, PhysicsAgent, CompositionAgent
)
from .tools import GeneratedScript, ToolCall

logger = logging.getLogger(__name__)

# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

class CodeGenerationOrchestrator:
    """
    Main orchestrator that coordinates the MoE system to generate complete Python scripts.
    This implements the full architectural blueprint from the PDF.
    """
    
    def __init__(self):
        """Initialize all expert agents."""
        logger.info("🎭 Initializing CodeGenerationOrchestrator...")
        
        try:
            # Initialize expert agents
            self.conductor = ConductorAgent()
            self.particle_agent = ParticleCreationAgent()
            self.color_agent = ColorPaletteAgent()
            self.motion_agent = MotionDynamicsAgent()
            self.physics_agent = PhysicsAgent()
            self.composition_agent = CompositionAgent()
            
            logger.info("✅ All agents initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize agents: {e}")
            raise

    async def generate_script(self, user_request: str) -> GeneratedScript:
        """
        Main entry point: generate a complete Tölvera script from user request.
        
        Args:
            user_request: Natural language request like "move a blue pixel from left to right"
            
        Returns:
            GeneratedScript with complete Python code
        """
        try:
            logger.info(f"🎯 Processing request: {user_request}")
            
            # Step 1: Conductor creates execution plan
            logger.info("📋 Step 1: Planning with ConductorAgent...")
            plan = await self.conductor.plan_task(user_request)
            logger.info(f"📋 Plan created: {plan.description}")
            
            # Step 2: Execute plan through expert agents
            logger.info("⚡ Step 2: Executing plan through expert agents...")
            all_tool_calls = []
            context = {}  # Track context between agents
            
            for i, step in enumerate(plan.steps):
                logger.info(f"⚡ Step {i+1}/{len(plan.steps)}: {step}")
                
                # Route to appropriate expert based on step content
                if self._is_particle_task(step):
                    result = await self.particle_agent.execute_task(step)
                    all_tool_calls.extend(result.tool_calls)
                    # Update context with particle info
                    self._update_context_from_tool_calls(context, result.tool_calls)
                
                elif self._is_color_task(step):
                    result = await self.color_agent.execute_task(step, context)
                    all_tool_calls.extend(result.tool_calls)
                
                elif self._is_motion_task(step):
                    result = await self.motion_agent.execute_task(step, context)
                    all_tool_calls.extend(result.tool_calls)
                
                elif self._is_physics_task(step):
                    result = await self.physics_agent.execute_task(step, context)
                    all_tool_calls.extend(result.tool_calls)
                
                elif self._is_composition_task(step):
                    # This is handled at the end, skip for now
                    logger.info("📝 Composition step noted, will handle at end")
                    continue
                
                else:
                    logger.warning(f"❓ Unrecognized step type: {step}")
            
            # Step 3: CompositionAgent assembles final script
            logger.info("📝 Step 3: Assembling final script with CompositionAgent...")
            script = await self.composition_agent.compose_script(user_request, all_tool_calls)
            
            logger.info(f"✅ Successfully generated script: {script.title}")
            logger.info(f"🔧 Used {len(all_tool_calls)} tool calls")
            
            return script
            
        except Exception as e:
            logger.error(f"❌ Script generation failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Return fallback script
            fallback_script = GeneratedScript(
                title=f"Fallback Script for: {user_request}",
                code=self._generate_emergency_fallback(user_request),
                explanation=f"Generated emergency fallback due to error: {str(e)}"
            )
            logger.warning("⚠️ Returning emergency fallback script")
            return fallback_script

    def _update_context_from_tool_calls(self, context: Dict[str, Any], tool_calls: List[ToolCall]):
        """Update context with information from tool calls."""
        for call in tool_calls:
            if call.tool_name == "create_particles":
                context["particles_created"] = call.parameters.get("n", 1)
                context["species_id"] = call.parameters.get("species_id", 0)
                context["num_species"] = max(context.get("num_species", 1), context["species_id"] + 1)

    def _is_particle_task(self, step: str) -> bool:
        """Check if step involves particle creation."""
        keywords = ["create", "particle", "pixel", "spawn", "generate", "position", "place"]
        return any(word in step.lower() for word in keywords)

    def _is_color_task(self, step: str) -> bool:
        """Check if step involves color operations.""" 
        keywords = ["color", "blue", "red", "green", "yellow", "paint", "tint", "hue", "shade"]
        return any(word in step.lower() for word in keywords)

    def _is_motion_task(self, step: str) -> bool:
        """Check if step involves movement."""
        keywords = ["move", "velocity", "motion", "right", "left", "up", "down", "speed", "varying", "direction"]
        return any(word in step.lower() for word in keywords)

    def _is_physics_task(self, step: str) -> bool:
        """Check if step involves complex physics."""
        keywords = ["flock", "physics", "gravity", "force", "attract", "repel", "behavior", "emergent", "interaction"]
        return any(word in step.lower() for word in keywords)

    def _is_composition_task(self, step: str) -> bool:
        """Check if step involves final script assembly."""
        keywords = ["assemble", "compose", "script", "final", "combine", "generate", "code", "file"]
        return any(word in step.lower() for word in keywords)

    def _generate_emergency_fallback(self, user_request: str) -> str:
        """Generate an emergency fallback script when everything fails."""
        return f'''"""
Emergency fallback script for: {user_request}
This is a minimal Tölvera script generated when the MoE system encountered errors.
"""

import taichi as ti
from tolvera import Tolvera, run

def main(**kwargs):
    """
    Emergency fallback implementation.
    Creates a basic particle system as requested: {user_request}
    """
    tv = Tolvera(n=50, species=1, **kwargs)
    
    @ti.kernel
    def init_particles():
        """Initialize basic particles."""
        for i in range(min(10, tv.pn)):
            tv.p.field[i].active = 1.0
            tv.p.field[i].species = 0
            tv.p.field[i].pos = [tv.x * 0.2, tv.y * 0.5]  # Start position
            tv.p.field[i].vel = [1.0, 0.0]  # Basic rightward movement
            tv.p.field[i].size = 8.0
            tv.p.field[i].mass = 1.0
    
    @ti.kernel
    def update_particles():
        """Update particle movement."""
        for i in range(tv.pn):
            if tv.p.field[i].active > 0:
                # Move particles
                tv.p.field[i].pos += tv.p.field[i].vel
                
                # Simple boundary wrapping
                if tv.p.field[i].pos[0] > tv.x:
                    tv.p.field[i].pos[0] = 0
                if tv.p.field[i].pos[0] < 0:
                    tv.p.field[i].pos[0] = tv.x
    
    # Set basic blue color
    tv.s.species.field[0].rgba = [0.2, 0.4, 1.0, 1.0]
    
    # Initialize
    init_particles()
    
    @tv.render
    def _():
        # Dark background
        tv.px.background(0.05, 0.05, 0.1)
        
        # Update and render
        update_particles()
        tv.px.particles(tv.p, tv.s.species(), "circle")
        
        return tv.px

if __name__ == '__main__':
    try:
        run(main)
    except KeyboardInterrupt:
        print("\\nExiting emergency fallback script.")
'''

    async def generate_multiple_scripts(self, requests: List[str]) -> List[GeneratedScript]:
        """Generate multiple scripts from a list of requests."""
        scripts = []
        
        for i, request in enumerate(requests):
            logger.info(f"🔄 Processing request {i+1}/{len(requests)}: {request}")
            try:
                script = await self.generate_script(request)
                scripts.append(script)
            except Exception as e:
                logger.error(f"❌ Failed to generate script for '{request}': {e}")
                # Add fallback
                fallback = GeneratedScript(
                    title=f"Failed: {request}",
                    code=self._generate_emergency_fallback(request),
                    explanation=f"Fallback due to error: {str(e)}"
                )
                scripts.append(fallback)
        
        return scripts

    def get_agent_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status information about all agents."""
        return {
            "conductor": {
                "model": self.conductor.model_name,
                "role": "Master planner and task decomposition",
                "status": "ready"
            },
            "particle": {
                "model": self.particle_agent.model_name,
                "role": "Particle creation and positioning",
                "status": "ready"
            },
            "color": {
                "model": self.color_agent.model_name,
                "role": "Color management",
                "status": "ready"
            },
            "motion": {
                "model": self.motion_agent.model_name,
                "role": "Movement and velocity",
                "status": "ready"
            },
            "physics": {
                "model": self.physics_agent.model_name,
                "role": "Complex physics interactions",
                "status": "ready"
            },
            "composition": {
                "model": self.composition_agent.model_name,
                "role": "Final script assembly",
                "status": "ready"
            }
        }

# =============================================================================
# SIMPLIFIED ORCHESTRATOR FOR TESTING
# =============================================================================

class SimpleOrchestrator:
    """
    Simplified orchestrator for testing and development.
    Uses fewer agents and simplified logic.
    """
    
    def __init__(self):
        """Initialize with minimal agents."""
        logger.info("🔧 Initializing SimpleOrchestrator for testing...")
        
        try:
            self.conductor = ConductorAgent()
            self.composition_agent = CompositionAgent()
            logger.info("✅ Simple orchestrator initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize simple orchestrator: {e}")
            raise

    async def generate_script(self, user_request: str) -> GeneratedScript:
        """Generate script with simplified logic."""
        try:
            logger.info(f"🎯 Simple processing: {user_request}")
            
            # Create a basic plan
            plan = await self.conductor.plan_task(user_request)
            
            # Generate some basic tool calls based on the request
            tool_calls = self._generate_basic_tool_calls(user_request)
            
            # Compose script
            script = await self.composition_agent.compose_script(user_request, tool_calls)
            
            return script
            
        except Exception as e:
            logger.error(f"❌ Simple generation failed: {e}")
            return GeneratedScript(
                title=f"Simple Fallback: {user_request}",
                code=self._generate_emergency_fallback(user_request),
                explanation=f"Simple fallback due to error: {str(e)}"
            )

    def _generate_basic_tool_calls(self, user_request: str) -> List[ToolCall]:
        """Generate basic tool calls based on keywords in the request."""
        tool_calls = []
        request_lower = user_request.lower()
        
        # Always create some particles
        n_particles = 3 if "three" in request_lower else 1
        position = [0.1, 0.5] if "left" in request_lower else None
        
        tool_calls.append(ToolCall(
            tool_name="create_particles",
            parameters={"n": n_particles, "species_id": 0, "position": position}
        ))
        
        # Handle colors
        if "blue" in request_lower:
            tool_calls.append(ToolCall(
                tool_name="set_species_color",
                parameters={"species_id": 0, "color": [0.0, 0.0, 1.0, 1.0]}
            ))
        elif "red" in request_lower:
            tool_calls.append(ToolCall(
                tool_name="set_species_color",
                parameters={"species_id": 0, "color": [1.0, 0.0, 0.0, 1.0]}
            ))
        elif "green" in request_lower:
            tool_calls.append(ToolCall(
                tool_name="set_species_color",
                parameters={"species_id": 0, "color": [0.0, 1.0, 0.0, 1.0]}
            ))
        
        # Handle movement
        if "right" in request_lower:
            tool_calls.append(ToolCall(
                tool_name="set_species_velocity",
                parameters={"species_id": 0, "velocity": [2.0, 0.0]}
            ))
        elif "left" in request_lower and "move" in request_lower:
            tool_calls.append(ToolCall(
                tool_name="set_species_velocity", 
                parameters={"species_id": 0, "velocity": [-2.0, 0.0]}
            ))
        elif "up" in request_lower:
            tool_calls.append(ToolCall(
                tool_name="set_species_velocity",
                parameters={"species_id": 0, "velocity": [0.0, -2.0]}
            ))
        elif "down" in request_lower:
            tool_calls.append(ToolCall(
                tool_name="set_species_velocity",
                parameters={"species_id": 0, "velocity": [0.0, 2.0]}
            ))
        
        # Handle varying speeds
        if "varying" in request_lower or "different" in request_lower:
            tool_calls.append(ToolCall(
                tool_name="apply_varying_speeds",
                parameters={"species_id": 0, "base_speed": 2.0, "variation": 1.0}
            ))
        
        return tool_calls

    def _generate_emergency_fallback(self, user_request: str) -> str:
        """Generate emergency fallback (reuse from main orchestrator)."""
        return CodeGenerationOrchestrator()._generate_emergency_fallback(user_request)

# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_orchestrator(simple: bool = False) -> CodeGenerationOrchestrator:
    """
    Factory function to create the appropriate orchestrator.
    
    Args:
        simple: If True, creates SimpleOrchestrator for testing
        
    Returns:
        Orchestrator instance
    """
    if simple:
        return SimpleOrchestrator()
    else:
        return CodeGenerationOrchestrator()