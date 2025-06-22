# src/tolvera/llm/compositional/tools.py
"""
Tool definitions and Pydantic schemas for the MoE system.
These define the "Lego bricks" that agents can use to compose Tölvera sketches.

Based on Table 1 from the architectural blueprint PDF.
"""

from typing import List, Optional, Tuple
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
# AGENT COMMUNICATION SCHEMAS
# =============================================================================

class ToolCall(BaseModel):
    """Represents a tool call to execute."""
    tool_name: str = Field(description="Tool function name")
    parameters: dict = Field(description="Tool parameters")

class TaskResult(BaseModel):
    """Result from an expert agent."""
    tool_calls: List[ToolCall] = Field(description="Tool calls to execute")
    explanation: str = Field(description="What was accomplished")

class TaskPlan(BaseModel):
    """Execution plan from conductor."""
    description: str = Field(description="Overall goal")
    steps: List[str] = Field(description="Step-by-step breakdown")

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