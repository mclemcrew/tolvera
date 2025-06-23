# src/tolvera/llm/compositional/orchestrator.py
"""
Fixed orchestrator with proper agent routing and better error handling.
COMPLETELY ENHANCED VERSION with number extraction and physics behavior detection.
"""

import logging
import re
from typing import List, Dict, Any
from .agents import (
    RobustConductorAgent, RobustParticleCreationAgent, RobustColorPaletteAgent,
    RobustMotionDynamicsAgent, RobustPhysicsAgent, RobustCompositionAgent,
    tool_calls_to_python_code_enhanced
)
from .tools import GeneratedScript, ToolCall

logger = logging.getLogger(__name__)

class RobustCodeGenerationOrchestrator:
    """
    COMPLETELY ENHANCED orchestrator with proper agent routing, number extraction, 
    and comprehensive fallback generation.
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
        """Generate a complete Tölvera script with enhanced agent routing and fallbacks."""
        
        try:
            logger.info(f"🎯 Processing request: {user_request}")
            
            # Step 1: Create execution plan
            logger.info("📋 Step 1: Planning with RobustConductorAgent...")
            plan = await self.conductor.plan_task(user_request)
            logger.info(f"📋 Plan created: {plan.description}")
            
            # Step 2: Execute plan through expert agents WITH ENHANCED ROUTING
            logger.info("⚡ Step 2: Executing plan through expert agents...")
            all_tool_calls = []
            context = {}
            
            for i, step in enumerate(plan.steps):
                logger.info(f"⚡ Step {i+1}/{len(plan.steps)}: {step}")
                
                # ENHANCED: Route to appropriate expert based on step content
                try:
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
            
            # Step 3: Generate ENHANCED fallback tool calls if we don't have any
            if not all_tool_calls:
                logger.warning("⚠️ No tool calls generated, creating enhanced fallbacks based on request")
                all_tool_calls = self._generate_enhanced_fallback_tool_calls(user_request)
            
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
        ENHANCED: Much better agent routing with comprehensive keyword detection.
        """
        step_lower = step.lower()
        
        # Physics keywords (check first - most specific)
        physics_keywords = [
            "bounc", "collision", "physics", "gravity", "force", "attract", "repel",
            "flock", "swarm", "around", "random", "chaotic", "turbulent",
            "float", "drift", "wander", "spiral", "orbit", "oscillat"
        ]
        
        # Motion keywords
        motion_keywords = [
            "move", "velocity", "speed", "motion", "direction", "travel",
            "right", "left", "up", "down", "horizontal", "vertical",
            "diagonal", "straight", "curve", "path"
        ]
        
        # Color keywords  
        color_keywords = [
            "color", "colour", "blue", "red", "green", "yellow", "white", "black",
            "cyan", "magenta", "orange", "purple", "pink", "rainbow", "bright"
        ]
        
        # Particle creation keywords
        particle_keywords = [
            "create", "spawn", "generate", "add", "make", "particles", "particle",
            "pixels", "pixel", "dots", "points", "circles", "balls"
        ]
        
        # Check in order of priority
        if any(keyword in step_lower for keyword in physics_keywords):
            return "physics"
        elif any(keyword in step_lower for keyword in motion_keywords):
            return "motion" 
        elif any(keyword in step_lower for keyword in color_keywords):
            return "color"
        elif any(keyword in step_lower for keyword in particle_keywords):
            return "particle"
        else:
            # Default routing for unclear steps
            if "step 1" in step_lower or "first" in step_lower:
                return "particle"
            elif "step 2" in step_lower or "second" in step_lower:
                return "color"  
            elif "step 3" in step_lower or "third" in step_lower:
                return "physics"
            else:
                return "particle"  # Ultimate fallback

    def _generate_enhanced_fallback_tool_calls(self, user_request: str) -> List[ToolCall]:
        """ENHANCED: Much better fallback generation with comprehensive extraction."""
        logger.info("🔧 Generating enhanced fallback tool calls from request analysis")
        
        tool_calls = []
        request_lower = user_request.lower()
        
        # ENHANCED: Extract numbers more accurately
        n_particles = self._extract_particle_count(request_lower)
        
        # Determine position based on request
        position = self._extract_position(request_lower)
        
        # Create particles with extracted count
        tool_calls.append(ToolCall(
            tool_name="create_particles",
            parameters={"n": n_particles, "species_id": 0, "position": position}
        ))
        
        # Extract and apply colors
        color = self._extract_color(request_lower)
        if color:
            tool_calls.append(ToolCall(
                tool_name="set_species_color",
                parameters={"species_id": 0, "color": color}
            ))
        
        # Extract and apply physics behaviors
        physics_behavior = self._extract_physics_behavior(request_lower)
        if physics_behavior:
            tool_calls.append(physics_behavior)
        
        # Extract and apply basic movement if no physics
        elif not physics_behavior:
            velocity = self._extract_velocity(request_lower)
            if velocity:
                tool_calls.append(ToolCall(
                    tool_name="set_species_velocity",
                    parameters={"species_id": 0, "velocity": velocity}
                ))
        
        logger.info(f"🔧 Generated {len(tool_calls)} enhanced fallback tool calls")
        return tool_calls
    
    def _extract_particle_count(self, request: str) -> int:
        """Extract particle count from natural language with comprehensive number detection."""
        
        # Number word mapping
        number_words = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "a": 1, "single": 1, "couple": 2, "few": 3, "several": 5,
            "many": 8, "lots": 10
        }
        
        # Check for number words first
        for word, num in number_words.items():
            if f" {word} " in f" {request} " or request.startswith(word):
                return num
        
        # Check for digits
        numbers = re.findall(r'\b(\d+)\b', request)
        if numbers:
            return max(1, min(20, int(numbers[0])))  # Clamp between 1-20
        
        # Default based on plurality and context
        if "particle" in request and "particles" not in request:
            return 1
        elif "pixel" in request and "pixels" not in request:
            return 1
        elif "circle" in request and "circles" not in request:
            return 1
        else:
            return 3  # Default for plural forms
    
    def _extract_color(self, request: str) -> List[float]:
        """Extract color from natural language with comprehensive color detection."""
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
            "cyan": [0.0, 1.0, 1.0, 1.0],
            "magenta": [1.0, 0.0, 1.0, 1.0],
            "lime": [0.5, 1.0, 0.0, 1.0],
            "navy": [0.0, 0.0, 0.5, 1.0]
        }
        
        for color_name, color_value in color_map.items():
            if color_name in request:
                return color_value
        return None
    
    def _extract_physics_behavior(self, request: str) -> ToolCall:
        """Extract physics behavior from natural language with comprehensive behavior detection."""
        if "bounc" in request or "around" in request:
            return ToolCall(
                tool_name="apply_bouncing_behavior",
                parameters={"species_id": 0, "speed_range": [1.0, 3.0], "collision_mode": "bounce"}
            )
        elif "float" in request or "drift" in request or "gentle" in request:
            return ToolCall(
                tool_name="apply_gentle_movement", 
                parameters={"species_id": 0, "speed": 0.8}
            )
        elif "random" in request or "chaotic" in request:
            return ToolCall(
                tool_name="apply_random_movement",
                parameters={"species_id": 0, "speed": 2.0, "randomness": 0.8}
            )
        elif "spiral" in request or "orbit" in request:
            return ToolCall(
                tool_name="apply_rotational_movement",
                parameters={"species_id": 0, "angular_speed": 0.05}
            )
        return None
    
    def _extract_velocity(self, request: str) -> List[float]:
        """Extract basic velocity from movement descriptions."""
        if "right" in request and ("move" in request or "moving" in request):
            return [2.0, 0.0]
        elif "left" in request and ("move" in request or "moving" in request):
            return [-2.0, 0.0]
        elif ("up" in request or "upward" in request) and ("move" in request or "moving" in request):
            return [0.0, -2.0]
        elif ("down" in request or "downward" in request) and ("move" in request or "moving" in request):
            return [0.0, 2.0]
        elif "top" in request and "bottom" in request:
            return [0.0, 2.0]  # Top to bottom
        elif "left" in request and "right" in request:
            return [2.0, 0.0]  # Left to right
        return None
    
    def _extract_position(self, request: str) -> List[float]:
        """Extract position from natural language with comprehensive position detection."""
        if "left" in request and "right" not in request:
            return [0.1, 0.5]
        elif "right" in request and "left" not in request:
            return [0.9, 0.5]
        elif "top" in request and "bottom" not in request:
            return [0.5, 0.1]
        elif "bottom" in request and "top" not in request:
            return [0.5, 0.9]
        elif "center" in request or "centre" in request or "middle" in request:
            return [0.5, 0.5]
        return None  # Random placement

    def _update_context_from_tool_calls(self, context: Dict[str, Any], tool_calls: List[ToolCall]):
        """Update context with information from tool calls."""
        for call in tool_calls:
            if call.tool_name == "create_particles":
                context["particles_created"] = call.parameters.get("n", 1)
                context["species_id"] = call.parameters.get("species_id", 0)
                context["num_species"] = max(context.get("num_species", 1), context["species_id"] + 1)

    def _generate_ultimate_fallback(self, user_request: str) -> str:
        """Generate the ultimate fallback script when everything fails."""
        
        # Extract basic info for fallback
        n_particles = self._extract_particle_count(user_request.lower())
        color = self._extract_color(user_request.lower()) or [0.0, 1.0, 0.0, 1.0]
        has_bouncing = "bounc" in user_request.lower() or "around" in user_request.lower()
        
        if has_bouncing:
            physics_code = """
                # Bouncing physics
                if tv.p.field[i].pos[0] <= 0 or tv.p.field[i].pos[0] >= tv.x:
                    tv.p.field[i].vel[0] *= -1.0
                    tv.p.field[i].pos[0] = ti.max(0.0, ti.min(tv.x, tv.p.field[i].pos[0]))
                
                if tv.p.field[i].pos[1] <= 0 or tv.p.field[i].pos[1] >= tv.y:
                    tv.p.field[i].vel[1] *= -1.0
                    tv.p.field[i].pos[1] = ti.max(0.0, ti.min(tv.y, tv.p.field[i].pos[1]))"""
        else:
            physics_code = """
                # Basic boundary wrapping
                if tv.p.field[i].pos[0] > tv.x:
                    tv.p.field[i].pos[0] = 0
                if tv.p.field[i].pos[0] < 0:
                    tv.p.field[i].pos[0] = tv.x"""
        
        return f'''"""
Ultimate fallback script for: {user_request}
This script is generated when the entire MoE system fails.
"""

import taichi as ti
from tolvera import Tolvera, run

def main(**kwargs):
    """
    Ultimate fallback implementation.
    Enhanced particle system for: {user_request}
    """
    tv = Tolvera(n={n_particles}, species=1, **kwargs)
    
    @ti.kernel
    def init_particles():
        """Initialize particles with enhanced setup."""
        # Deactivate all particles first
        for i in range(tv.pn):
            tv.p.field[i].active = 0.0
        
        # Initialize {n_particles} particles
        for i in range({n_particles}):
            tv.p.field[i].active = 1.0
            tv.p.field[i].species = 0
            tv.p.field[i].pos = ti.Vector([tv.x * ti.random(), tv.y * ti.random()])
            # Random velocities for dynamic motion
            speed = 1.0 + ti.random() * 2.0
            angle = ti.random() * 2.0 * 3.14159
            tv.p.field[i].vel = ti.Vector([speed * ti.cos(angle), speed * ti.sin(angle)])
            tv.p.field[i].size = 10.0 + ti.random() * 6.0
            tv.p.field[i].mass = 1.0
    
    @ti.kernel
    def update_particles():
        """Enhanced particle movement."""
        for i in range({n_particles}):
            if tv.p.field[i].active > 0:
                tv.p.field[i].pos += tv.p.field[i].vel
                {physics_code}
    
    @ti.kernel
    def draw_particles():
        """Draw enhanced particles."""
        tv.px.background(0.0, 0.0, 0.0)
        
        for i in range({n_particles}):
            if tv.p.field[i].active > 0:
                pos = tv.p.field[i].pos
                x = ti.cast(pos[0], ti.i32)
                y = ti.cast(pos[1], ti.i32)
                size = ti.cast(tv.p.field[i].size, ti.i32)
                
                species_id = tv.p.field[i].species
                color = tv.s.species.field[species_id].rgba
                
                tv.px.circle(x, y, size, color, fill=1)
    
    # Set enhanced color
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
                "role": "Enhanced master planner with number and behavior extraction",
                "status": "ready"
            },
            "particle": {
                "model": self.particle_agent.model_name,
                "role": "Enhanced particle creation with accurate count extraction",
                "status": "ready"
            },
            "color": {
                "model": self.color_agent.model_name,
                "role": "Color management with comprehensive color detection",
                "status": "ready"
            },
            "motion": {
                "model": self.motion_agent.model_name,
                "role": "Movement with directional detection",
                "status": "ready"
            },
            "physics": {
                "model": self.physics_agent.model_name,
                "role": "Enhanced physics with bouncing and behavior detection",
                "status": "ready"
            },
            "composition": {
                "model": self.composition_agent.model_name,
                "role": "Enhanced script assembly with physics behavior support",
                "status": "ready"
            }
        }

def create_robust_orchestrator() -> RobustCodeGenerationOrchestrator:
    """Create the enhanced robust orchestrator."""
    return RobustCodeGenerationOrchestrator()