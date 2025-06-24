# src/tolvera/llm/compositional/orchestrator.py
"""
IMPROVED orchestrator with better agent routing and organic behavior generation.
"""

import logging
import re
from typing import List, Dict, Any
from .agents import (
    RobustConductorAgent, RobustParticleCreationAgent, RobustColorPaletteAgent,
    RobustMotionDynamicsAgent, RobustPhysicsAgent, RobustCompositionAgent
)
from .tools import GeneratedScript, ToolCall

logger = logging.getLogger(__name__)

class RobustCodeGenerationOrchestrator:
    """
    IMPROVED orchestrator with better agent routing and more organic particle behaviors.
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
        """Generate a complete Tölvera script with improved agent routing."""
        
        try:
            logger.info(f"🎯 Processing request: {user_request}")
            
            # Step 1: Create execution plan
            logger.info("📋 Step 1: Planning with RobustConductorAgent...")
            plan = await self.conductor.plan_task(user_request)
            logger.info(f"📋 Plan created: {plan.description}")
            
            # Step 2: Execute plan through expert agents (they return ToolCall objects, NOT JSON)
            logger.info("⚡ Step 2: Executing plan through expert agents...")
            all_tool_calls = []  # This will contain ToolCall objects from individual agents
            context = {"particles_created": 0, "colors_set": False, "motion_applied": False, "physics_applied": False}
            
            for i, step in enumerate(plan.steps):
                logger.info(f"⚡ Step {i+1}/{len(plan.steps)}: {step}")
                
                # IMPROVED: Better agent routing logic
                try:
                    agent_type = self._determine_agent_type(step)
                    
                    if agent_type == "particle":
                        result = await self.particle_agent.execute_task(step)
                        all_tool_calls.extend(result.tool_calls)
                        self._update_context_from_tool_calls(context, result.tool_calls)
                    
                    elif agent_type == "color":
                        result = await self.color_agent.execute_task(step, context)
                        all_tool_calls.extend(result.tool_calls)
                        context["colors_set"] = True
                    
                    elif agent_type == "motion":
                        result = await self.motion_agent.execute_task(step, context)
                        all_tool_calls.extend(result.tool_calls)
                        context["motion_applied"] = True
                    
                    elif agent_type == "physics":
                        result = await self.physics_agent.execute_task(step, context)
                        all_tool_calls.extend(result.tool_calls)
                        context["physics_applied"] = True
                    
                    elif agent_type == "composition":
                        logger.info("📝 Composition step - will handle at end")
                        continue
                    
                    else:
                        logger.warning(f"❓ Unrecognized step type for: {step}")
                        # Apply intelligent fallbacks based on missing components
                        fallback_calls = self._generate_step_fallback(step, context)
                        all_tool_calls.extend(fallback_calls)
                
                except Exception as e:
                    logger.warning(f"⚠️ Step {i+1} failed: {e}, applying fallbacks")
                    fallback_calls = self._generate_step_fallback(step, context)
                    all_tool_calls.extend(fallback_calls)
            
            # IMPROVED: Check for missing essential components and add them
            logger.info("🔧 Checking for missing essential components...")
            missing_components = self._detect_missing_components(user_request, context, all_tool_calls)
            if missing_components:
                logger.info(f"🔧 Adding missing components: {missing_components}")
                additional_calls = await self._generate_missing_components(user_request, missing_components, context)
                all_tool_calls.extend(additional_calls)
            
            # Step 3: Ensure we have minimum viable tool calls
            if not all_tool_calls or context["particles_created"] == 0:
                logger.warning("⚠️ Insufficient tool calls generated, creating enhanced fallbacks")
                fallback_calls = self._generate_enhanced_fallback_tool_calls(user_request)
                all_tool_calls.extend(fallback_calls)
            
            # Step 4: CompositionAgent assembles final script (converts ToolCalls to Python code)
            logger.info("📝 Step 3: Assembling final script with RobustCompositionAgent...")
            # NOTE: This returns a GeneratedScript with Python code, NOT JSON
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

    def _detect_missing_components(self, user_request: str, context: Dict[str, Any], tool_calls: List[ToolCall]) -> List[str]:
        """Detect missing essential components based on the request and context."""
        missing = []
        request_lower = user_request.lower()
        
        # Check for movement requirements
        movement_keywords = ["move", "moving", "travel", "right", "left", "up", "down", "direction"]
        has_movement_request = any(keyword in request_lower for keyword in movement_keywords)
        has_motion_tool = any(call.tool_name in ["set_species_velocity", "set_particle_velocity"] for call in tool_calls)
        
        if has_movement_request and not has_motion_tool and not context.get("motion_applied", False):
            missing.append("motion")
        
        # Check for physics requirements  
        physics_keywords = ["bounce", "bouncing", "around", "flock", "physics", "behavior"]
        has_physics_request = any(keyword in request_lower for keyword in physics_keywords)
        has_physics_tool = any("apply_" in call.tool_name for call in tool_calls)
        
        if has_physics_request and not has_physics_tool and not context.get("physics_applied", False):
            missing.append("physics")
        
        return missing
    
    async def _generate_missing_components(self, user_request: str, missing_components: List[str], context: Dict[str, Any]) -> List[ToolCall]:
        """Generate missing components using the appropriate agents."""
        additional_calls = []
        
        for component in missing_components:
            try:
                if component == "motion":
                    # Generate motion step from request
                    motion_step = self._generate_motion_step_from_request(user_request)
                    result = await self.motion_agent.execute_task(motion_step, context)
                    additional_calls.extend(result.tool_calls)
                    logger.info(f"🔧 Added missing motion: {result.explanation}")
                    
                elif component == "physics":
                    # Generate physics step from request  
                    physics_step = self._generate_physics_step_from_request(user_request)
                    result = await self.physics_agent.execute_task(physics_step, context)
                    additional_calls.extend(result.tool_calls)
                    logger.info(f"🔧 Added missing physics: {result.explanation}")
                    
            except Exception as e:
                logger.warning(f"⚠️ Failed to generate missing {component}: {e}")
        
        return additional_calls
    
    def _generate_motion_step_from_request(self, user_request: str) -> str:
        """Generate a motion step description from the user request."""
        request_lower = user_request.lower()
        
        if "right" in request_lower:
            return "Apply rightward movement to particles"
        elif "left" in request_lower:
            return "Apply leftward movement to particles"  
        elif "up" in request_lower:
            return "Apply upward movement to particles"
        elif "down" in request_lower:
            return "Apply downward movement to particles"
        elif "move" in request_lower or "moving" in request_lower:
            return "Apply movement to particles based on request"
        else:
            return "Apply default movement to particles"

    def _generate_physics_step_from_request(self, user_request: str) -> str:
        """Generate a physics step description from the user request."""
        request_lower = user_request.lower()
        
        if "bounce" in request_lower or "bouncing" in request_lower:
            return "Apply bouncing physics behavior to particles"
        elif "around" in request_lower:
            return "Apply movement physics for particles moving around"
        elif "flock" in request_lower:
            return "Apply flocking behavior to particles"
        else:
            return "Apply appropriate physics behavior to particles"

    def _determine_agent_type(self, step: str) -> str:
        """
        IMPROVED: Much better agent routing with clear priority system.
        """
        step_lower = step.lower()
        
        # Composition keywords (highest priority)
        composition_keywords = ["assemble", "final script", "compose", "generate script"]
        if any(keyword in step_lower for keyword in composition_keywords):
            return "composition"
        
        # Physics keywords (check before motion to avoid conflicts)
        physics_keywords = [
            "bounc", "collision", "physics", "flock", "birds", "behavior",
            "cohesion", "separation", "alignment", "emergent"
        ]
        if any(keyword in step_lower for keyword in physics_keywords):
            return "physics"
        
        # Motion keywords
        motion_keywords = [
            "move", "velocity", "speed", "motion", "direction", "travel",
            "right", "left", "up", "down", "constant velocity"
        ]
        if any(keyword in step_lower for keyword in motion_keywords):
            return "motion"
        
        # Color keywords  
        color_keywords = [
            "color", "colour", "blue", "red", "green", "yellow", "white", "black",
            "cyan", "magenta", "orange", "purple", "pink"
        ]
        if any(keyword in step_lower for keyword in color_keywords):
            return "color"
        
        # Particle creation keywords
        particle_keywords = [
            "create", "spawn", "generate", "add", "make", "particles", "particle",
            "pixels", "pixel", "positioned"
        ]
        if any(keyword in step_lower for keyword in particle_keywords):
            return "particle"
        
        # Default fallback based on step number
        if "1" in step or "first" in step_lower:
            return "particle"
        elif "2" in step or "second" in step_lower:
            return "color"
        elif "3" in step or "third" in step_lower:
            return "physics"
        elif "4" in step or "fourth" in step_lower or "final" in step_lower:
            return "composition"
        else:
            return "particle"  # Safe default

    def _generate_step_fallback(self, step: str, context: Dict[str, Any]) -> List[ToolCall]:
        """Generate intelligent fallbacks based on the step and context."""
        fallback_calls = []
        step_lower = step.lower()
        
        # If no particles created yet, create some
        if context["particles_created"] == 0 and ("particle" in step_lower or "create" in step_lower):
            n_particles = self._extract_particle_count(step_lower)
            fallback_calls.append(ToolCall(
                tool_name="create_particles",
                parameters={"n": n_particles, "species_id": 0, "position": None}
            ))
            context["particles_created"] = n_particles
        
        # If colors mentioned but not set
        if not context["colors_set"] and any(color in step_lower for color in ["blue", "red", "green", "yellow"]):
            color = self._extract_color(step_lower)
            if color:
                fallback_calls.append(ToolCall(
                    tool_name="set_species_color",
                    parameters={"species_id": 0, "color": color}
                ))
                context["colors_set"] = True
        
        # If physics mentioned but not applied
        if not context["physics_applied"] and any(word in step_lower for word in ["bounce", "flock", "physics"]):
            if "bounc" in step_lower:
                fallback_calls.append(ToolCall(
                    tool_name="apply_bouncing_behavior",
                    parameters={"species_id": 0, "speed_range": [1.0, 3.0], "collision_mode": "bounce"}
                ))
            elif "flock" in step_lower:
                fallback_calls.append(ToolCall(
                    tool_name="apply_flock_behavior",
                    parameters={"species_id": 0, "cohesion": 0.8, "separation": 0.6, "alignment": 0.7}
                ))
            context["physics_applied"] = True
        
        return fallback_calls

    def _generate_enhanced_fallback_tool_calls(self, user_request: str) -> List[ToolCall]:
        """Generate comprehensive fallback tool calls."""
        logger.info("🔧 Generating enhanced fallback tool calls")
        
        tool_calls = []
        request_lower = user_request.lower()
        
        # Extract particle count
        n_particles = self._extract_particle_count(request_lower)
        
        # Create particles
        tool_calls.append(ToolCall(
            tool_name="create_particles",
            parameters={"n": n_particles, "species_id": 0, "position": None}
        ))
        
        # Extract and apply colors
        color = self._extract_color(request_lower)
        if color:
            tool_calls.append(ToolCall(
                tool_name="set_species_color",
                parameters={"species_id": 0, "color": color}
            ))
        
        # Extract and apply physics behaviors
        if "bounc" in request_lower or "around" in request_lower:
            tool_calls.append(ToolCall(
                tool_name="apply_bouncing_behavior",
                parameters={"species_id": 0, "speed_range": [1.0, 3.0], "collision_mode": "bounce"}
            ))
        elif "flock" in request_lower or "birds" in request_lower:
            tool_calls.append(ToolCall(
                tool_name="apply_flock_behavior",
                parameters={"species_id": 0, "cohesion": 0.8, "separation": 0.6, "alignment": 0.7}
            ))
        else:
            # Apply basic movement
            velocity = self._extract_velocity(request_lower)
            if velocity:
                tool_calls.append(ToolCall(
                    tool_name="set_species_velocity",
                    parameters={"species_id": 0, "velocity": velocity}
                ))
        
        logger.info(f"🔧 Generated {len(tool_calls)} enhanced fallback tool calls")
        return tool_calls
    
    def _extract_particle_count(self, request: str) -> int:
        """Extract particle count from natural language."""
        number_words = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "a": 1, "single": 1, "couple": 2, "few": 3, "several": 5
        }
        
        for word, num in number_words.items():
            if f" {word} " in f" {request} " or request.startswith(word):
                return num
        
        numbers = re.findall(r'\b(\d+)\b', request)
        if numbers:
            return max(1, min(20, int(numbers[0])))
        
        # Default based on singular/plural
        if any(word in request for word in ["particle", "pixel", "circle"]) and not any(word in request for word in ["particles", "pixels", "circles"]):
            return 1
        else:
            return 3
    
    def _extract_color(self, request: str) -> List[float]:
        """Extract color from natural language."""
        color_map = {
            "red": [1.0, 0.0, 0.0, 1.0],
            "green": [0.0, 1.0, 0.0, 1.0],
            "blue": [0.0, 0.0, 1.0, 1.0],
            "yellow": [1.0, 1.0, 0.0, 1.0],
            "orange": [1.0, 0.5, 0.0, 1.0],
            "purple": [0.5, 0.0, 1.0, 1.0],
            "white": [1.0, 1.0, 1.0, 1.0],
        }
        
        for color_name, color_value in color_map.items():
            if color_name in request:
                return color_value
        return None
    
    def _extract_velocity(self, request: str) -> List[float]:
        """Extract velocity from movement descriptions."""
        if "right" in request or "left to right" in request:
            return [2.0, 0.0]
        elif "left" in request:
            return [-2.0, 0.0]
        elif "up" in request or "upward" in request:
            return [0.0, -2.0]
        elif "down" in request or "downward" in request or "top to bottom" in request:
            return [0.0, 2.0]
        return None

    def _update_context_from_tool_calls(self, context: Dict[str, Any], tool_calls: List[ToolCall]):
        """Update context with information from tool calls."""
        for call in tool_calls:
            if call.tool_name == "create_particles":
                context["particles_created"] = call.parameters.get("n", 1)

    def _generate_ultimate_fallback(self, user_request: str) -> str:
        """Generate the ultimate fallback script when everything fails."""
        
        n_particles = self._extract_particle_count(user_request.lower())
        color = self._extract_color(user_request.lower()) or [0.0, 1.0, 0.0, 1.0]
        has_bouncing = "bounc" in user_request.lower()
        
        return f'''"""
Ultimate fallback script for: {user_request}
This script is generated when the MoE system fails.
"""

import taichi as ti
from tolvera import Tolvera, run

def main(**kwargs):
    """
    Ultimate fallback implementation for: {user_request}
    """
    tv = Tolvera(n={n_particles}, species=1, **kwargs)
    
    @ti.kernel
    def init_particles():
        for i in range(tv.pn):
            tv.p.field[i].active = 0.0
        
        for i in range({n_particles}):
            tv.p.field[i].active = 1.0
            tv.p.field[i].species = 0
            tv.p.field[i].pos = ti.Vector([tv.x * ti.random(), tv.y * ti.random()])
            speed = 1.0 + ti.random() * 2.0
            angle = ti.random() * 2.0 * 3.14159
            tv.p.field[i].vel = ti.Vector([speed * ti.cos(angle), speed * ti.sin(angle)])
            tv.p.field[i].size = 10.0 + ti.random() * 6.0
            tv.p.field[i].mass = 1.0
    
    @ti.kernel
    def update_particles():
        for i in range({n_particles}):
            if tv.p.field[i].active > 0:
                tv.p.field[i].pos += tv.p.field[i].vel
                {"# Bouncing physics" if has_bouncing else "# Basic wrapping"}
                if tv.p.field[i].pos[0] <= 0 or tv.p.field[i].pos[0] >= tv.x:
                    tv.p.field[i].vel[0] *= {"-1.0" if has_bouncing else "1.0; tv.p.field[i].pos[0] = tv.x if tv.p.field[i].pos[0] < 0 else 0.0"}
                if tv.p.field[i].pos[1] <= 0 or tv.p.field[i].pos[1] >= tv.y:
                    tv.p.field[i].vel[1] *= {"-1.0" if has_bouncing else "1.0; tv.p.field[i].pos[1] = tv.y if tv.p.field[i].pos[1] < 0 else 0.0"}
    
    @ti.kernel
    def draw_particles():
        tv.px.background(0.0, 0.0, 0.0)
        for i in range({n_particles}):
            if tv.p.field[i].active > 0:
                pos = tv.p.field[i].pos
                x = ti.cast(pos[0], ti.i32)
                y = ti.cast(pos[1], ti.i32)
                size = ti.cast(tv.p.field[i].size, ti.i32)
                tv.px.circle(x, y, size, ti.Vector({color}), fill=1)
    
    tv.s.species.field[0].rgba = ti.Vector({color})
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
        print("\\nExiting ultimate fallback script.")
'''

def create_robust_orchestrator() -> RobustCodeGenerationOrchestrator:
    """Create the improved robust orchestrator."""
    return RobustCodeGenerationOrchestrator()