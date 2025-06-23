# src/tolvera/llm/compositional/tools.py
"""
Tool definitions and Pydantic schemas for the MoE system.
Fixed with classic typing annotations for pydantic-ai compatibility.
"""

from typing import List, Optional, Tuple, Dict, Any  # Classic typing imports
from pydantic import BaseModel, Field

# =============================================================================
# TOOL SCHEMAS - Define the structured data formats for agent communication
# =============================================================================

class CreateParticlesSchema(BaseModel):
    """Create new particles in the system."""
    n: int = Field(description="Number of particles to create", ge=1, le=5000)
    species_id: int = Field(description="The integer ID of the species for these particles", ge=0)
    position: Optional[Tuple[float, float]] = Field(
        description="Optional initial (x, y) position as normalized coordinates (0.0-1.0)",
        default=None
    )

class SetParticleVelocitySchema(BaseModel):
    """Set velocity for specific particles."""
    particle_ids: List[int] = Field(description="List of particle IDs to modify")
    velocity: Tuple[float, float] = Field(description="The (vx, vy) velocity vector to apply")

class SetSpeciesVelocitySchema(BaseModel):
    """Set velocity for all particles of a species."""
    species_id: int = Field(description="Species ID to modify")
    velocity: Tuple[float, float] = Field(description="The (vx, vy) velocity vector to apply")

class SetParticleColorSchema(BaseModel):
    """Set color for specific particles."""
    particle_ids: List[int] = Field(description="List of particle IDs to modify")
    color: Tuple[float, float, float, float] = Field(description="The (r, g, b, a) color (0.0-1.0)")

class SetSpeciesColorSchema(BaseModel):
    """Set color for all particles of a species."""
    species_id: int = Field(description="Species ID to modify")
    color: Tuple[float, float, float, float] = Field(description="The (r, g, b, a) color (0.0-1.0)")

class FillBackgroundSchema(BaseModel):
    """Fill the screen with a color."""
    color: Tuple[float, float, float, float] = Field(description="The (r, g, b, a) color to fill screen with")

class SetGlobalStateSchema(BaseModel):
    """Set a global variable in the state system."""
    name: str = Field(description="The name of the global state variable")
    value: float = Field(description="The floating-point value to set")

class ApplyFlockBehaviorSchema(BaseModel):
    """Apply flocking behavior to a species."""
    species_id: int = Field(description="The species ID to apply this behavior to")
    cohesion: float = Field(description="Cohesion strength", ge=0.0, le=1.0)
    separation: float = Field(description="Separation strength", ge=0.0, le=1.0)
    alignment: float = Field(description="Alignment strength", ge=0.0, le=1.0)

class ApplyNoiseFieldSchema(BaseModel):
    """Apply a noise field force."""
    species_id: int = Field(description="Species to apply noise to")
    strength: float = Field(description="Noise strength", ge=0.0, le=1.0)
    scale: float = Field(description="Noise scale", ge=0.1, le=2.0)
    speed: float = Field(description="Noise evolution speed", ge=0.1, le=2.0)

class ApplyVaryingSpeedsSchema(BaseModel):
    """Apply varying speeds to particles of a species."""
    species_id: int = Field(description="Species to apply varying speeds to")
    base_speed: float = Field(description="Base speed magnitude", ge=0.0)
    variation: float = Field(description="Speed variation amount", ge=0.0)

# =============================================================================
# AGENT COMMUNICATION SCHEMAS - FIXED WITH CLASSIC TYPING
# =============================================================================

class ToolCall(BaseModel):
    """Represents a tool call to execute."""
    tool_name: str = Field(description="Tool function name")
    parameters: Dict[str, Any] = Field(description="Tool parameters")  # More specific than dict

class TaskResult(BaseModel):
    """Result from an expert agent."""
    tool_calls: List[ToolCall] = Field(description="Tool calls to execute")  # FIXED: List[ToolCall]
    explanation: str = Field(description="What was accomplished")

class TaskPlan(BaseModel):
    """Execution plan from conductor."""
    description: str = Field(description="Overall goal")
    steps: List[str] = Field(description="Step-by-step breakdown")  # FIXED: List[str]

class GeneratedScript(BaseModel):
    """Complete generated Tölvera script."""
    title: str = Field(description="Brief title describing what the script does")
    code: str = Field(description="Complete Python script code")
    explanation: str = Field(description="Explanation of how the script works")

# =============================================================================
# STANDARD COLOR DEFINITIONS
# =============================================================================

STANDARD_COLORS = {
    "red": [1.0, 0.0, 0.0, 1.0],
    "green": [0.0, 1.0, 0.0, 1.0],
    "blue": [0.0, 0.0, 1.0, 1.0],
    "yellow": [1.0, 1.0, 0.0, 1.0],
    "white": [1.0, 1.0, 1.0, 1.0],
    "black": [0.0, 0.0, 0.0, 1.0],
    "cyan": [0.0, 1.0, 1.0, 1.0],
    "magenta": [1.0, 0.0, 1.0, 1.0],
    "orange": [1.0, 0.5, 0.0, 1.0],
    "purple": [0.5, 0.0, 1.0, 1.0],
    "pink": [1.0, 0.5, 0.8, 1.0],
    "gray": [0.5, 0.5, 0.5, 1.0],
    "brown": [0.6, 0.3, 0.1, 1.0],
    "navy": [0.0, 0.0, 0.5, 1.0],
    "lime": [0.5, 1.0, 0.0, 1.0],
}

# =============================================================================
# TOOL REGISTRY - Maps tool names to their schemas
# =============================================================================

TOOL_REGISTRY = {
    "create_particles": CreateParticlesSchema,
    "set_particle_velocity": SetParticleVelocitySchema,
    "set_species_velocity": SetSpeciesVelocitySchema,
    "set_particle_color": SetParticleColorSchema,
    "set_species_color": SetSpeciesColorSchema,
    "fill_background": FillBackgroundSchema,
    "set_global_state": SetGlobalStateSchema,
    "apply_flock_behavior": ApplyFlockBehaviorSchema,
    "apply_noise_field": ApplyNoiseFieldSchema,
    "apply_varying_speeds": ApplyVaryingSpeedsSchema,
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_color_by_name(color_name: str) -> List[float]:
    """Get RGBA color values by name."""
    color_name = color_name.lower().strip()
    return STANDARD_COLORS.get(color_name, STANDARD_COLORS["white"])

def validate_tool_call(tool_call: ToolCall) -> bool:
    """Validate that a tool call uses the correct schema."""
    if tool_call.tool_name not in TOOL_REGISTRY:
        return False
    
    try:
        schema_class = TOOL_REGISTRY[tool_call.tool_name]
        schema_class(**tool_call.parameters)
        return True
    except Exception:
        return False

def normalize_position(x: float, y: float) -> Tuple[float, float]:
    """Normalize position coordinates to 0.0-1.0 range."""
    return (max(0.0, min(1.0, x)), max(0.0, min(1.0, y)))

# =============================================================================
# TEMPLATE GENERATION HELPERS
# =============================================================================

def generate_tolvera_imports() -> str:
    """Generate standard Tölvera imports."""
    return '''"""
{title}
"""

import taichi as ti
from tolvera import Tolvera, run'''

def generate_main_function_template(num_particles: int = 100, num_species: int = 1) -> str:
    """Generate main function template."""
    return f'''
def main(**kwargs):
    """Main function to run the Tölvera simulation."""
    tv = Tolvera(n={num_particles}, species={num_species}, **kwargs)
    
    # Initialize simulation state
    @ti.kernel
    def init_simulation():
        # Initialization code will be inserted here
        pass
    
    # Update simulation each frame
    @ti.kernel  
    def update_simulation():
        # Update code will be inserted here
        pass
    
    # Run initialization
    init_simulation()
    
    @tv.render
    def _():
        # Clear background
        tv.px.background(0.0, 0.0, 0.0)
        
        # Update simulation
        update_simulation()
        
        # Render particles
        tv.px.particles(tv.p, tv.s.species(), "circle")
        
        return tv.px

if __name__ == '__main__':
    try:
        run(main)
    except KeyboardInterrupt:
        print("\\nExiting.")'''

# =============================================================================
# SCRIPT GENERATION HELPERS
# =============================================================================

# src/tolvera/llm/compositional/tools.py - FIXED VERSION

# src/tolvera/llm/compositional/tools.py - CORRECTED VERSION

# src/tolvera/llm/compositional/tools.py - FINAL CORRECTED VERSION

def tool_calls_to_python_code(tool_calls: List[ToolCall], user_request: str) -> str:
    """
    Generate Tölvera scripts that match the exact patterns from working examples.
    FINAL FIX: Proper particle management, correct positioning, clean rendering.
    """
    
    # Analyze tool calls
    particles_created = 1  # Default to 1
    num_species = 1
    species_colors = {}
    species_velocities = {}
    positions = {}
    
    # Detect circular motion from request
    is_circular_motion = "circle" in user_request.lower() and "moving" in user_request.lower()
    
    for call in tool_calls:
        if call.tool_name == "create_particles":
            n = call.parameters.get("n", 1)
            species_id = call.parameters.get("species_id", 0)
            position = call.parameters.get("position")
            
            particles_created = n  # Use exact count
            num_species = max(num_species, species_id + 1)
            positions[species_id] = position
        
        elif call.tool_name == "set_species_color":
            species_id = call.parameters.get("species_id", 0)
            color = call.parameters.get("color", [1.0, 1.0, 1.0, 1.0])
            species_colors[species_id] = color
        
        elif call.tool_name == "set_species_velocity":
            species_id = call.parameters.get("species_id", 0)
            velocity = call.parameters.get("velocity", [0.0, 0.0])
            species_velocities[species_id] = velocity
    
    # Generate code based on motion type
    if is_circular_motion:
        # Circular motion implementation
        position = positions.get(0, [0.5, 0.5])  # Default to center
        color = species_colors.get(0, [1.0, 0.0, 0.0, 1.0])  # Default red
        
        script = f'''"""
{user_request}
Generated by Tölvera MoE system.
"""

import taichi as ti
from tolvera import Tolvera, run

def main(**kwargs):
    """
    {user_request}
    """
    tv = Tolvera(n={particles_created}, species={num_species}, **kwargs)

    # Fields for circular motion
    center = ti.Vector.field(2, dtype=ti.f32, shape=())
    radius = ti.field(dtype=ti.f32, shape=())
    angle = ti.field(dtype=ti.f32, shape=())
    speed = ti.field(dtype=ti.f32, shape=())

    @ti.kernel
    def init_particles():
        """Initialize particles following the working examples pattern."""
        # FIXED: Proper center positioning
        center[None] = ti.Vector([tv.x * {position[0]}, tv.y * {position[1]}])
        radius[None] = min(tv.x, tv.y) * 0.2  # Reasonable radius
        angle[None] = 0.0
        speed[None] = 0.03

        # FIXED: Deactivate ALL particles first (like working examples)
        for i in range(tv.pn):
            tv.p.field[i].active = 0.0

        # FIXED: Only activate and setup the particles we actually want
        for i in range({particles_created}):
            tv.p.field[i].pos = center[None] + radius[None] * ti.Vector([ti.cos(angle[None]), ti.sin(angle[None])])
            tv.p.field[i].vel = ti.Vector([0.0, 0.0])
            tv.p.field[i].active = 1.0
            tv.p.field[i].species = 0
            tv.p.field[i].size = 16.0
            tv.p.field[i].mass = 1.0

    @ti.kernel
    def update_particles():
        """Update particle positions for circular motion."""
        angle[None] += speed[None]
        
        # FIXED: Only update the particles we created
        for i in range({particles_created}):
            if tv.p.field[i].active > 0:
                new_pos_x = center[None][0] + radius[None] * ti.cos(angle[None])
                new_pos_y = center[None][1] + radius[None] * ti.sin(angle[None])
                tv.p.field[i].pos = ti.Vector([new_pos_x, new_pos_y])

    @ti.kernel
    def draw_particles():
        """Draw only the particles we created."""
        tv.px.background(0.0, 0.0, 0.0)
        
        # FIXED: Only draw the particles we actually created
        for i in range({particles_created}):
            if tv.p.field[i].active > 0:
                pos = tv.p.field[i].pos
                x = ti.cast(pos[0], ti.i32)
                y = ti.cast(pos[1], ti.i32)
                size = ti.cast(tv.p.field[i].size, ti.i32)
                
                species_id = tv.p.field[i].species
                color = tv.s.species.field[species_id].rgba
                
                tv.px.circle(x, y, size, color, fill=1)

    # Set colors in Python scope
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
    
    else:
        # Linear motion implementation
        init_lines = []
        update_lines = []
        
        # Build initialization for each particle
        init_lines.append("        # FIXED: Deactivate ALL particles first")
        init_lines.append("        for i in range(tv.pn):")
        init_lines.append("            tv.p.field[i].active = 0.0")
        init_lines.append("")
        
        for i in range(particles_created):
            species_id = 0  # Default species
            position = positions.get(species_id, [0.1, 0.5])
            velocity = species_velocities.get(species_id, [2.0, 0.0])
            
            init_lines.extend([
                f"        # Initialize particle {i}",
                f"        tv.p.field[{i}].pos = ti.Vector([tv.x * {position[0]}, tv.y * {position[1]}])",
                f"        tv.p.field[{i}].vel = ti.Vector({velocity})",
                f"        tv.p.field[{i}].active = 1.0",
                f"        tv.p.field[{i}].species = {species_id}",
                f"        tv.p.field[{i}].size = 16.0",
                f"        tv.p.field[{i}].mass = 1.0",
                ""
            ])
        
        # Add movement update if velocities exist
        if species_velocities:
            update_lines.extend([
                f"        # Update movement for {particles_created} particles",
                f"        for i in range({particles_created}):",
                "            if tv.p.field[i].active > 0:",
                "                tv.p.field[i].pos += tv.p.field[i].vel",
                "                # Boundary wrapping",
                "                if tv.p.field[i].pos[0] > tv.x:",
                "                    tv.p.field[i].pos[0] = 0.0",
                "                if tv.p.field[i].pos[0] < 0:",
                "                    tv.p.field[i].pos[0] = tv.x",
                "                if tv.p.field[i].pos[1] > tv.y:",
                "                    tv.p.field[i].pos[1] = 0.0",
                "                if tv.p.field[i].pos[1] < 0:",
                "                    tv.p.field[i].pos[1] = tv.y"
            ])
        
        init_code = "\n".join(init_lines)
        update_code = "\n".join(update_lines) if update_lines else "        pass"
        
        # Color setup
        color = species_colors.get(0, [0.0, 0.0, 1.0, 1.0])  # Default blue
        
        script = f'''"""
{user_request}
Generated by Tölvera MoE system.
"""

import taichi as ti
from tolvera import Tolvera, run

def main(**kwargs):
    """
    {user_request}
    """
    tv = Tolvera(n={particles_created}, species={num_species}, **kwargs)

    @ti.kernel
    def init_particles():
        """Initialize particles following the working examples pattern."""
{init_code}

    @ti.kernel
    def update_particles():
        """Update particle positions."""
{update_code}

    @ti.kernel
    def draw_particles():
        """Draw only the particles we created."""
        tv.px.background(0.0, 0.0, 0.0)
        
        # FIXED: Only draw the particles we actually created
        for i in range({particles_created}):
            if tv.p.field[i].active > 0:
                pos = tv.p.field[i].pos
                x = ti.cast(pos[0], ti.i32)
                y = ti.cast(pos[1], ti.i32)
                size = ti.cast(tv.p.field[i].size, ti.i32)
                
                species_id = tv.p.field[i].species
                color = tv.s.species.field[species_id].rgba
                
                tv.px.circle(x, y, size, color, fill=1)

    # Set colors in Python scope
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
    
    return script
# =============================================================================
# EXPERT AGENT TOOL DESCRIPTIONS
# =============================================================================

PARTICLE_AGENT_TOOLS = """
Available tools:
- create_particles(n, species_id, position): Create n particles of specified species
  - n: number of particles (1-5000)
  - species_id: which species (0-5)
  - position: optional (x,y) normalized coordinates (0.0-1.0)

Position reference:
- Left side: x=0.1, Center: x=0.5, Right: x=0.9
- Top: y=0.1, Middle: y=0.5, Bottom: y=0.9
"""

COLOR_AGENT_TOOLS = """
Available tools:
- set_particle_color(particle_ids, color): Set RGBA color for specific particles
- set_species_color(species_id, color): Set color for entire species

Standard colors (use exact values):
- red: [1.0, 0.0, 0.0, 1.0]
- green: [0.0, 1.0, 0.0, 1.0]
- blue: [0.0, 0.0, 1.0, 1.0]
- yellow: [1.0, 1.0, 0.0, 1.0]
- white: [1.0, 1.0, 1.0, 1.0]
- black: [0.0, 0.0, 0.0, 1.0]
"""

MOTION_AGENT_TOOLS = """
Available tools:
- set_particle_velocity(particle_ids, velocity): Set velocity for specific particles
- set_species_velocity(species_id, velocity): Set velocity for entire species
- apply_varying_speeds(species_id, base_speed, variation): Apply varying speeds

Movement guidelines:
- Slow movement: magnitude ~0.5-2.0
- Medium movement: magnitude ~2.0-5.0
- Fast movement: magnitude ~5.0-10.0

Directions:
- Right: [2.0, 0.0] (positive x)
- Left: [-2.0, 0.0] (negative x)  
- Up: [0.0, -2.0] (negative y)
- Down: [0.0, 2.0] (positive y)
"""

PHYSICS_AGENT_TOOLS = """
Available tools:
- apply_flock_behavior(species_id, cohesion, separation, alignment): Configure flocking
- apply_noise_field(species_id, strength, scale, speed): Add random forces
- set_global_state(name, value): Set physics parameters

Focus on multi-particle behaviors, forces, and emergent systems.
For simple single-particle movement, defer to MotionDynamicsAgent.
"""