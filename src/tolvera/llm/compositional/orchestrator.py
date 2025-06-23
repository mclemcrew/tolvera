# src/tolvera/llm/compositional/orchestrator.py
"""
Fixed orchestrator with proper agent routing and better error handling.
"""

import logging
from typing import List, Dict, Any
from .agents import (
    RobustConductorAgent, RobustParticleCreationAgent, RobustColorPaletteAgent,
    RobustMotionDynamicsAgent, RobustPhysicsAgent, RobustCompositionAgent
)
from .tools import GeneratedScript, ToolCall

logger = logging.getLogger(__name__)

class RobustCodeGenerationOrchestrator:
    """
    Fixed orchestrator with proper agent routing and enhanced error handling.
    """
    
    def __init__(self):
        """Initialize all expert agents."""
        logger.info("🎭 Initializing RobustCodeGenerationOrchestrator...")
        
        try:
            self.conductor = RobustConductorAgent()
            self.particle_agent = RobustParticleCreationAgent()
            self.color_agent = RobustColorPaletteAgent()
            self.motion_agent = RobustMotionDynamicsAgent()
            self.physics_agent = RobustPhysicsAgent()
            self.composition_agent = RobustCompositionAgent()
            
            logger.info("✅ All robust agents initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize robust agents: {e}")
            raise

    async def generate_script(self, user_request: str) -> GeneratedScript:
        """Generate a complete Tölvera script with proper agent routing."""
        
        try:
            logger.info(f"🎯 Processing request: {user_request}")
            
            # Step 1: Create execution plan
            logger.info("📋 Step 1: Planning with RobustConductorAgent...")
            plan = await self.conductor.plan_task(user_request)
            logger.info(f"📋 Plan created: {plan.description}")
            
            # Step 2: Execute plan through expert agents WITH PROPER ROUTING
            logger.info("⚡ Step 2: Executing plan through expert agents...")
            all_tool_calls = []
            context = {}
            
            for i, step in enumerate(plan.steps):
                logger.info(f"⚡ Step {i+1}/{len(plan.steps)}: {step}")
                
                # FIXED: Route to appropriate expert based on step content
                try:
                    # Determine the correct agent for this step
                    agent_type = self._determine_agent_type(step)
                    
                    if agent_type == "particle":
                        result = await self.particle_agent.execute_task(step)
                        all_tool_calls.extend(result.tool_calls)
                        self._update_context_from_tool_calls(context, result.tool_calls)
                    
                    elif agent_type == "color":
                        result = await self.color_agent.execute_task(step, context)
                        all_tool_calls.extend(result.tool_calls)
                    
                    elif agent_type == "motion":
                        result = await self.motion_agent.execute_task(step, context)
                        all_tool_calls.extend(result.tool_calls)
                    
                    elif agent_type == "physics":
                        result = await self.physics_agent.execute_task(step, context)
                        all_tool_calls.extend(result.tool_calls)
                    
                    elif agent_type == "composition":
                        logger.info("📝 Composition step noted, will handle at end")
                        continue
                    
                    else:
                        logger.warning(f"❓ Unrecognized step type for: {step}")
                        # Try particle agent as fallback
                        result = await self.particle_agent.execute_task(step)
                        all_tool_calls.extend(result.tool_calls)
                
                except Exception as e:
                    logger.warning(f"⚠️ Step {i+1} failed: {e}, continuing with fallbacks")
                    # Continue with other steps even if one fails
            
            # Step 3: Generate fallback tool calls if we don't have any
            if not all_tool_calls:
                logger.warning("⚠️ No tool calls generated, creating fallbacks based on request")
                all_tool_calls = self._generate_fallback_tool_calls(user_request)
            
            # Step 4: CompositionAgent assembles final script
            logger.info("📝 Step 3: Assembling final script with RobustCompositionAgent...")
            script = await self.composition_agent.compose_script(user_request, all_tool_calls)
            
            logger.info(f"✅ Successfully generated script: {script.title}")
            logger.info(f"🔧 Used {len(all_tool_calls)} tool calls")
            
            return script
            
        except Exception as e:
            logger.error(f"❌ Script generation failed: {e}")
            
            # Ultimate fallback script
            fallback_script = GeneratedScript(
                title=f"Ultimate Fallback: {user_request}",
                code=self._generate_ultimate_fallback(user_request),
                explanation=f"Generated ultimate fallback due to complete system failure: {str(e)}"
            )
            logger.warning("⚠️ Returning ultimate fallback script")
            return fallback_script

    def _determine_agent_type(self, step: str) -> str:
        """
        FIXED: Properly determine which agent should handle each step.
        This is the key fix for routing issues.
        """
        step_lower = step.lower()
        
        # Particle creation keywords (highest priority)
        particle_keywords = [
            "create", "spawn", "generate", "add", "make particles", "make particle",
            "positioned", "position", "place", "put particles"
        ]
        
        # Color keywords 
        color_keywords = [
            "color", "colour", "blue", "red", "green", "yellow", "white", "black",
            "cyan", "magenta", "orange", "purple", "pink", "set color", "apply color",
            "paint", "tint", "hue"
        ]
        
        # Motion keywords
        motion_keywords = [
            "move", "velocity", "speed", "motion", "direction", "apply velocity",
            "set velocity", "right", "left", "up", "down", "horizontal", "vertical",
            "acceleration", "deceleration"
        ]
        
        # Physics keywords
        physics_keywords = [
            "flock", "flocking", "physics", "gravity", "force", "attract", "repel", 
            "behavior", "behaviour", "emergent", "swarm", "birds", "cohesion", 
            "separation", "alignment", "bounce", "bouncing", "elasticity"
        ]
        
        # Composition keywords
        composition_keywords = [
            "assemble", "compose", "script", "final", "combine", "generate script",
            "visualization", "complete"
        ]
        
        # Check in order of specificity
        # 1. Check for composition first (most specific)
        if any(keyword in step_lower for keyword in composition_keywords):
            return "composition"
        
        # 2. Check for physics (specific behaviors)
        if any(keyword in step_lower for keyword in physics_keywords):
            return "physics"
        
        # 3. Check for motion (before color, as motion words are more specific)
        if any(keyword in step_lower for keyword in motion_keywords):
            return "motion"
        
        # 4. Check for color (before particle, as color is more specific)
        if any(keyword in step_lower for keyword in color_keywords):
            return "color"
        
        # 5. Check for particle creation (most general)
        if any(keyword in step_lower for keyword in particle_keywords):
            return "particle"
        
        # Default to particle if unclear
        logger.warning(f"⚠️ Could not determine agent type for step: {step}")
        return "particle"

    def _generate_fallback_tool_calls(self, user_request: str) -> List[ToolCall]:
        """Generate basic tool calls based on keywords in the request."""
        logger.info("🔧 Generating fallback tool calls from request analysis")
        
        tool_calls = []
        request_lower = user_request.lower()
        
        # Extract number of particles
        n_particles = 1
        numbers = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
        for i, word in enumerate(numbers):
            if word in request_lower:
                n_particles = i + 1
                break
        
        # Check for explicit numbers
        import re
        numbers_found = re.findall(r'\b(\d+)\b', request_lower)
        if numbers_found:
            n_particles = max(n_particles, int(numbers_found[0]))
        
        # Determine position - use proper normalized coordinates
        position = None
        if "left" in request_lower:
            position = [0.1, 0.5]
        elif "right" in request_lower:
            position = [0.9, 0.5]
        elif "top" in request_lower:
            position = [0.5, 0.1]
        elif "bottom" in request_lower:
            position = [0.5, 0.9]
        elif "center" in request_lower or "centre" in request_lower:
            position = [0.5, 0.5]
        # Default to None for random positioning
        
        # Create particles
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
        elif "yellow" in request_lower:
            tool_calls.append(ToolCall(
                tool_name="set_species_color",
                parameters={"species_id": 0, "color": [1.0, 1.0, 0.0, 1.0]}
            ))
        
        # Handle movement - use simple literal values only
        velocity = None
        if "right" in request_lower and ("move" in request_lower or "moving" in request_lower):
            velocity = [2.0, 0.0]
        elif "left" in request_lower and ("move" in request_lower or "moving" in request_lower):
            velocity = [-2.0, 0.0]
        elif ("up" in request_lower or "upward" in request_lower) and ("move" in request_lower or "moving" in request_lower):
            velocity = [0.0, -2.0]
        elif ("down" in request_lower or "downward" in request_lower) and ("move" in request_lower or "moving" in request_lower):
            velocity = [0.0, 2.0]
        elif "top" in request_lower and "bottom" in request_lower:
            velocity = [0.0, 2.0]  # Top to bottom
        
        if velocity:
            tool_calls.append(ToolCall(
                tool_name="set_species_velocity",
                parameters={"species_id": 0, "velocity": velocity}
            ))
        
        # Handle flocking
        if any(word in request_lower for word in ["flock", "swarm", "birds", "together"]):
            tool_calls.append(ToolCall(
                tool_name="apply_flock_behavior",
                parameters={"species_id": 0, "cohesion": 0.7, "separation": 0.3, "alignment": 0.5}
            ))
        
        logger.info(f"🔧 Generated {len(tool_calls)} fallback tool calls")
        return tool_calls

    def _update_context_from_tool_calls(self, context: Dict[str, Any], tool_calls: List[ToolCall]):
        """Update context with information from tool calls."""
        for call in tool_calls:
            if call.tool_name == "create_particles":
                context["particles_created"] = call.parameters.get("n", 1)
                context["species_id"] = call.parameters.get("species_id", 0)
                context["num_species"] = max(context.get("num_species", 1), context["species_id"] + 1)

    def _generate_ultimate_fallback(self, user_request: str) -> str:
        """Generate the ultimate fallback script when everything fails."""
        return f'''"""
Ultimate fallback script for: {user_request}
This script is generated when the entire MoE system fails.
"""

import taichi as ti
from tolvera import Tolvera, run

def main(**kwargs):
    """
    Ultimate fallback implementation.
    Basic particle system for: {user_request}
    """
    tv = Tolvera(n=50, species=1, **kwargs)
    
    @ti.kernel
    def init_particles():
        """Initialize particles with basic setup."""
        for i in range(min(10, tv.pn)):
            tv.p.field[i].active = 1.0
            tv.p.field[i].species = 0
            tv.p.field[i].pos = [tv.x * 0.2, tv.y * 0.5]
            tv.p.field[i].vel = [1.0, 0.0]
            tv.p.field[i].size = 8.0
            tv.p.field[i].mass = 1.0
    
    @ti.kernel
    def update_particles():
        """Basic particle movement."""
        for i in range(tv.pn):
            if tv.p.field[i].active > 0:
                tv.p.field[i].pos += tv.p.field[i].vel
                
                # Boundary wrapping
                if tv.p.field[i].pos[0] > tv.x:
                    tv.p.field[i].pos[0] = 0
                if tv.p.field[i].pos[0] < 0:
                    tv.p.field[i].pos[0] = tv.x
    
    # Set basic color based on request
    color = [0.2, 0.4, 1.0, 1.0]  # Default blue
    if "red" in "{user_request}".lower():
        color = [1.0, 0.2, 0.2, 1.0]
    elif "green" in "{user_request}".lower():
        color = [0.2, 1.0, 0.2, 1.0]
    elif "yellow" in "{user_request}".lower():
        color = [1.0, 1.0, 0.2, 1.0]
    
    tv.s.species.field[0].rgba = color
    
    # Initialize
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
        print("\\nExiting ultimate fallback script.")
'''

    async def generate_multiple_scripts(self, requests: List[str]) -> List[GeneratedScript]:
        """Generate multiple scripts with robust error handling."""
        scripts = []
        
        for i, request in enumerate(requests):
            logger.info(f"🔄 Processing request {i+1}/{len(requests)}: {request}")
            try:
                script = await self.generate_script(request)
                scripts.append(script)
            except Exception as e:
                logger.error(f"❌ Failed to generate script for '{request}': {e}")
                # Add ultimate fallback
                fallback = GeneratedScript(
                    title=f"Failed: {request}",
                    code=self._generate_ultimate_fallback(request),
                    explanation=f"Ultimate fallback due to error: {str(e)}"
                )
                scripts.append(fallback)
        
        return scripts

    def get_agent_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status information about all agents."""
        return {
            "conductor": {
                "model": self.conductor.model_name,
                "role": "Master planner with robust error handling",
                "status": "ready"
            },
            "particle": {
                "model": self.particle_agent.model_name,
                "role": "Particle creation with fallbacks",
                "status": "ready"
            },
            "color": {
                "model": self.color_agent.model_name,
                "role": "Color management with fallbacks",
                "status": "ready"
            },
            "motion": {
                "model": self.motion_agent.model_name,
                "role": "Movement with fallbacks",
                "status": "ready"
            },
            "physics": {
                "model": self.physics_agent.model_name,
                "role": "Physics with fallbacks",
                "status": "ready"
            },
            "composition": {
                "model": self.composition_agent.model_name,
                "role": "Script assembly (direct generation)",
                "status": "ready"
            }
        }

def create_robust_orchestrator() -> RobustCodeGenerationOrchestrator:
    """Create the robust orchestrator."""
    return RobustCodeGenerationOrchestrator()