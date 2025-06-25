"""
Tool definitions and Pydantic schemas for the test MoE
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class ToolCall(BaseModel):
    """Enhanced tool call with both structured parameters and code snippets."""
    tool_name: str = Field(description="Tool function name")
    parameters: Dict[str, Any] = Field(description="Tool parameters", default_factory=dict)
    code_snippet: Optional[str] = Field(description="Python/Taichi code implementation", default="")
    explanation: Optional[str] = Field(description="What this code accomplishes", default="")

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

# Quick look up if we want for colors

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
# TOOL SCHEMAS - These were used when I was using more json output for the
# tools, but I had a lot of issues so a lot of this isn't used anymore.
# =============================================================================

# class CreateParticlesSchema(BaseModel):
#     """Create new particles in the system."""
#     n: int = Field(description="Number of particles to create", ge=1, le=5000)
#     species_id: int = Field(description="The integer ID of the species for these particles", ge=0)
#     position: Optional[Tuple[float, float]] = Field(
#         description="Optional initial (x, y) position as normalized coordinates (0.0-1.0)",
#         default=None
#     )

# class SetParticleVelocitySchema(BaseModel):
#     """Set velocity for specific particles."""
#     particle_ids: List[int] = Field(description="List of particle IDs to modify")
#     velocity: Tuple[float, float] = Field(description="The (vx, vy) velocity vector to apply")

# class SetSpeciesVelocitySchema(BaseModel):
#     """Set velocity for all particles of a species."""
#     species_id: int = Field(description="Species ID to modify")
#     velocity: Tuple[float, float] = Field(description="The (vx, vy) velocity vector to apply")

# class SetParticleColorSchema(BaseModel):
#     """Set color for specific particles."""
#     particle_ids: List[int] = Field(description="List of particle IDs to modify")
#     color: Tuple[float, float, float, float] = Field(description="The (r, g, b, a) color (0.0-1.0)")

# class SetSpeciesColorSchema(BaseModel):
#     """Set color for all particles of a species."""
#     species_id: int = Field(description="Species ID to modify")
#     color: Tuple[float, float, float, float] = Field(description="The (r, g, b, a) color (0.0-1.0)")

# class FillBackgroundSchema(BaseModel):
#     """Fill the screen with a color."""
#     color: Tuple[float, float, float, float] = Field(description="The (r, g, b, a) color to fill screen with")

# class SetGlobalStateSchema(BaseModel):
#     """Set a global variable in the state system."""
#     name: str = Field(description="The name of the global state variable")
#     value: float = Field(description="The floating-point value to set")

# class ApplyFlockBehaviorSchema(BaseModel):
#     """Apply flocking behavior to a species."""1
#     species_id: int = Field(description="The species ID to apply this behavior to")
#     cohesion: float = Field(description="Cohesion strength", ge=0.0, le=1.0)
#     separation: float = Field(description="Separation strength", ge=0.0, le=1.0)
#     alignment: float = Field(description="Alignment strength", ge=0.0, le=1.0)

# class ApplyBouncingBehaviorSchema(BaseModel):
#     """Apply bouncing physics behavior to particles."""
#     species_id: int = Field(description="Species to apply bouncing to")
#     speed_range: Tuple[float, float] = Field(description="Min and max speed range")
#     collision_mode: str = Field(description="Collision detection mode", default="bounce")

# class ApplyGentleMovementSchema(BaseModel):
#     """Apply gentle floating movement to particles."""
#     species_id: int = Field(description="Species to apply gentle movement to")
#     speed: float = Field(description="Movement speed", ge=0.0, le=5.0)

# class ApplyRandomMovementSchema(BaseModel):
#     """Apply random chaotic movement to particles."""
#     species_id: int = Field(description="Species to apply random movement to")
#     speed: float = Field(description="Movement speed", ge=0.0, le=5.0)
#     randomness: float = Field(description="Randomness factor", ge=0.0, le=1.0)

# class ApplyRotationalMovementSchema(BaseModel):
#     """Apply rotational/circular movement to particles."""
#     species_id: int = Field(description="Species to apply rotational movement to")
#     angular_speed: float = Field(description="Angular speed for rotation", ge=0.0, le=1.0)

# class ApplyNoiseFieldSchema(BaseModel):
#     """Apply a noise field force."""
#     species_id: int = Field(description="Species to apply noise to")
#     strength: float = Field(description="Noise strength", ge=0.0, le=1.0)
#     scale: float = Field(description="Noise scale", ge=0.1, le=2.0)
#     speed: float = Field(description="Noise evolution speed", ge=0.1, le=2.0)

# class ApplyVaryingSpeedsSchema(BaseModel):
#     """Apply varying speeds to particles of a species."""
#     species_id: int = Field(description="Species to apply varying speeds to")
#     base_speed: float = Field(description="Base speed magnitude", ge=0.0)
#     variation: float = Field(description="Speed variation amount", ge=0.0)

# TOOL_REGISTRY = {
#     "create_particles": CreateParticlesSchema,
#     "set_particle_velocity": SetParticleVelocitySchema,
#     "set_species_velocity": SetSpeciesVelocitySchema,
#     "set_particle_color": SetParticleColorSchema,
#     "set_species_color": SetSpeciesColorSchema,
#     "fill_background": FillBackgroundSchema,
#     "set_global_state": SetGlobalStateSchema,
#     "apply_flock_behavior": ApplyFlockBehaviorSchema,
#     "apply_bouncing_behavior": ApplyBouncingBehaviorSchema,
#     "apply_gentle_movement": ApplyGentleMovementSchema,
#     "apply_random_movement": ApplyRandomMovementSchema,
#     "apply_rotational_movement": ApplyRotationalMovementSchema,
#     "apply_noise_field": ApplyNoiseFieldSchema,
#     "apply_varying_speeds": ApplyVaryingSpeedsSchema,
# }

# PARTICLE_AGENT_TOOLS = """
# Available tools:
# - create_particles(n, species_id, position): Create n particles of specified species
#   - n: number of particles (1-5000)
#   - species_id: which species (0-5)
#   - position: optional (x,y) normalized coordinates (0.0-1.0)

# Position reference:
# - Left side: x=0.1, Center: x=0.5, Right: x=0.9
# - Top: y=0.1, Middle: y=0.5, Bottom: y=0.9
# """

# COLOR_AGENT_TOOLS = """
# Available tools:
# - set_particle_color(particle_ids, color): Set RGBA color for specific particles
# - set_species_color(species_id, color): Set color for entire species

# Standard colors (use exact values):
# - red: [1.0, 0.0, 0.0, 1.0]
# - green: [0.0, 1.0, 0.0, 1.0]
# - blue: [0.0, 0.0, 1.0, 1.0]
# - yellow: [1.0, 1.0, 0.0, 1.0]
# - white: [1.0, 1.0, 1.0, 1.0]
# - black: [0.0, 0.0, 0.0, 1.0]
# """

# MOTION_AGENT_TOOLS = """
# Available tools:
# - set_particle_velocity(particle_ids, velocity): Set velocity for specific particles
# - set_species_velocity(species_id, velocity): Set velocity for entire species
# - apply_varying_speeds(species_id, base_speed, variation): Apply varying speeds

# Movement guidelines:
# - Slow movement: magnitude ~0.5-2.0
# - Medium movement: magnitude ~2.0-5.0
# - Fast movement: magnitude ~5.0-10.0

# Directions:
# - Right: [2.0, 0.0] (positive x)
# - Left: [-2.0, 0.0] (negative x)  
# - Up: [0.0, -2.0] (negative y)
# - Down: [0.0, 2.0] (positive y)
# """

# PHYSICS_AGENT_TOOLS = """
# Available tools:
# - apply_bouncing_behavior(species_id, speed_range, collision_mode): Configure bouncing physics
# - apply_gentle_movement(species_id, speed): Apply gentle floating movement
# - apply_random_movement(species_id, speed, randomness): Apply chaotic random movement
# - apply_rotational_movement(species_id, angular_speed): Apply circular/orbital movement
# - apply_flock_behavior(species_id, cohesion, separation, alignment): Configure flocking
# - apply_noise_field(species_id, strength, scale, speed): Add random forces
# - set_global_state(name, value): Set physics parameters

# Focus on multi-particle behaviors, forces, and emergent systems.
# For simple single-particle movement, defer to MotionAgent.
# """