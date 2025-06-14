"""
Tölvera PydanticAI Models - Based on the Tölvera API 
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum

class ParticleShape(str, Enum):
    """Supported particle rendering shapes in Tölvera."""
    CIRCLE = "circle"
    POINT = "point"
    RECT = "rect"


class BackgroundBehavior(str, Enum):
    """Background rendering behaviors."""
    DIFFUSE = "diffuse"
    CLEAR = "clear"
    FADE = "fade"


# NEW: Support for multiple behaviors in one sketch
class BehaviorType(str, Enum):
    """Available Vera behaviors."""
    FLOCK = "flock"
    SLIME = "slime"
    PARTICLE_LIFE = "particle_life"


class ColorPalette(BaseModel):
    """Color palette configuration for the sketch."""
    background: Tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0),
        description="RGB background color (0.0-1.0)"
    )
    primary: Tuple[float, float, float] = Field(
        description="Primary particle color (0.0-1.0)"
    )
    secondary: Optional[Tuple[float, float, float]] = Field(
        default=None,
        description="Secondary particle color for multi-behavior sketches"
    )
    accent: Optional[Tuple[float, float, float]] = Field(
        default=None,
        description="Accent color for special effects"
    )

    @field_validator('background', 'primary', 'secondary', 'accent')
    @classmethod
    def validate_rgb(cls, v):
        if v is None:
            return v
        if not all(0.0 <= component <= 1.0 for component in v):
            raise ValueError("RGB values must be between 0.0 and 1.0")
        return v


class FlockBehaviorConfig(BaseModel):
    """Flocking behavior parameters for individual species."""
    separation_distance: float = Field(
        default=0.1, ge=0.0, le=1.0,
        description="Minimum distance between particles"
    )
    alignment_strength: float = Field(
        default=0.8, ge=0.0, le=2.0,
        description="Strength of velocity alignment"
    )
    cohesion_force: float = Field(
        default=0.6, ge=0.0, le=1.0,
        description="Attraction to local group center"
    )
    max_speed: float = Field(
        default=2.0, ge=0.1, le=10.0,
        description="Maximum particle velocity"
    )
    perception_radius: float = Field(
        default=50.0, ge=1.0, le=200.0,
        description="Distance for neighbor detection"
    )


class SlimeBehaviorConfig(BaseModel):
    """Slime mold behavior parameters for individual species."""
    sense_angle: float = Field(
        default=0.5, ge=0.1, le=1.57,
        description="Sensor angle in radians"
    )
    sense_dist: float = Field(
        default=20.0, ge=5.0, le=100.0,
        description="Sensing distance"
    )
    move_angle: float = Field(
        default=0.3, ge=0.1, le=1.57,
        description="Maximum turning angle"
    )
    move_step: float = Field(
        default=1.0, ge=0.1, le=5.0,
        description="Movement distance per step"
    )
    deposit_amount: float = Field(
        default=1.0, ge=0.1, le=2.0,
        description="Chemical deposit per step"
    )


class ParticleLifeConfig(BaseModel):
    """Particle life behavior parameters."""
    attraction_radius: float = Field(
        default=100.0, ge=10.0, le=300.0,
        description="Attraction radius"
    )
    repulsion_radius: float = Field(
        default=50.0, ge=5.0, le=100.0,
        description="Repulsion radius"
    )
    attraction_strength: float = Field(
        default=0.5, ge=0.0, le=2.0,
        description="Attraction force strength"
    )

class BehaviorInstance(BaseModel):
    """Configuration for a single behavior instance within a sketch."""
    behavior_type: BehaviorType = Field(description="Type of behavior")
    particle_count: int = Field(
        default=512, ge=50, le=5000,
        description="Number of particles for this behavior"
    )
    species_indices: List[int] = Field(
        default_factory=lambda: [0],
        description="Which species this behavior applies to"
    )
    
    # Behavior-specific configurations
    flock_config: Optional[FlockBehaviorConfig] = Field(
        default=None,
        description="Flocking parameters (if behavior_type is flock)"
    )
    slime_config: Optional[SlimeBehaviorConfig] = Field(
        default=None,
        description="Slime parameters (if behavior_type is slime)"
    )
    particle_life_config: Optional[ParticleLifeConfig] = Field(
        default=None,
        description="Particle life parameters (if behavior_type is particle_life)"
    )
    
    particle_shape: ParticleShape = Field(
        default=ParticleShape.CIRCLE,
        description="Particle shape for this behavior"
    )
    color: Optional[Tuple[float, float, float]] = Field(
        default=None,
        description="Custom color for this behavior's particles"
    )

    @model_validator(mode='after')
    def validate_behavior_config(self):
        """Verify the correct config is set based on behavior type."""
        if self.behavior_type == BehaviorType.FLOCK:
            if self.flock_config is None:
                self.flock_config = FlockBehaviorConfig()
            self.slime_config = None
            self.particle_life_config = None
        elif self.behavior_type == BehaviorType.SLIME:
            if self.slime_config is None:
                self.slime_config = SlimeBehaviorConfig()
            self.flock_config = None
            self.particle_life_config = None
        elif self.behavior_type == BehaviorType.PARTICLE_LIFE:
            if self.particle_life_config is None:
                self.particle_life_config = ParticleLifeConfig()
            self.flock_config = None
            self.slime_config = None
        
        return self


class RenderConfig(BaseModel):
    """Global rendering configuration for the entire sketch."""
    background_behavior: BackgroundBehavior = Field(
        default=BackgroundBehavior.DIFFUSE,
        description="How to handle background between frames"
    )
    diffuse_rate: float = Field(
        default=0.99, ge=0.0, le=1.0,
        description="Diffusion rate for trails (if using diffuse background)"
    )
    brightness: float = Field(
        default=1.0, ge=0.1, le=3.0,
        description="Overall brightness multiplier"
    )
    window_width: int = Field(
        default=1920, ge=400, le=3840,
        description="Canvas width"
    )
    window_height: int = Field(
        default=1080, ge=300, le=2160,
        description="Canvas height"
    )

class MultiBehaviorSketchConfig(BaseModel):
    """
    Configuration for Tölvera sketches that can combine multiple behaviors.
    
    This allows for complex compositions like flocking birds with slime mold trails,
    or particle life systems with flocking interactions.
    """
    
    sketch_name: str = Field(
        description="Name of the sketch",
        min_length=1, max_length=100
    )
    description: str = Field(
        description="Human-readable description of the artistic concept"
    )
    
    behaviors: List[BehaviorInstance] = Field(
        description="List of behaviors to combine in this sketch",
        min_length=1
    )
    
    render_config: RenderConfig = Field(
        default_factory=RenderConfig,
        description="Global rendering settings"
    )
    
    color_palette: ColorPalette = Field(
        description="Color scheme for the sketch"
    )
    
    global_speed: float = Field(
        default=1.0, ge=0.1, le=5.0,
        description="Global simulation speed multiplier"
    )
    substeps: int = Field(
        default=1, ge=1, le=10,
        description="Number of simulation substeps per frame"
    )

    @field_validator('sketch_name')
    @classmethod
    def validate_sketch_name(cls, v: str) -> str:
        """Ensure sketch name is filesystem-safe."""
        import re
        safe_name = re.sub(r'[^\w\-_]', '_', v)
        if not safe_name:
            raise ValueError("Sketch name cannot be empty after cleaning")
        return safe_name

    @model_validator(mode='after')
    def validate_behavior_composition(self):
        """Validate that the behavior combination makes sense."""
        total_particles = sum(b.particle_count for b in self.behaviors)
        if total_particles > 10000:
            raise ValueError(f"Total particle count ({total_particles}) too high - may cause performance issues")
        
        all_species = set()
        for behavior in self.behaviors:
            for species_idx in behavior.species_indices:
                if species_idx in all_species:
                    # This is fine since multiple behaviors can use the same species. I think?
                    pass
                all_species.add(species_idx)
        
        return self
    
    def get_total_particle_count(self) -> int:
        """Get total number of particles across all behaviors."""
        return sum(b.particle_count for b in self.behaviors)
    
    def get_max_species_index(self) -> int:
        """Get the highest species index used."""
        if not self.behaviors:
            return 0
        return max(max(b.species_indices) for b in self.behaviors)
    
    def get_behavior_summary(self) -> str:
        """Get human-readable summary of behaviors."""
        behavior_types = [b.behavior_type.value for b in self.behaviors]
        return f"Multi-behavior sketch with: {', '.join(behavior_types)}"
    
    def to_tolvera_kwargs(self) -> Dict[str, Any]:
        """Convert to kwargs for Tölvera constructor."""
        return {
            'particles': self.get_total_particle_count(),
            'species': self.get_max_species_index() + 1, 
            'x': self.render_config.window_width,
            'y': self.render_config.window_height,
            'speed': self.global_speed,
            'substep': self.substeps,
        }


# Response models for PydanticAI agents
class SketchGenerationResponse(BaseModel):
    config: MultiBehaviorSketchConfig = Field(description="Generated sketch configuration")
    python_code: str = Field(description="Generated Tölvera Python code")
    explanation: str = Field(description="Explanation of the artistic concept and implementation")
    suggestions: List[str] = Field(
        default_factory=list,
        description="Suggestions for further exploration"
    )
    performance_notes: List[str] = Field(
        default_factory=list,
        description="Performance considerations"
    )


class SketchModificationResponse(BaseModel):
    updated_config: MultiBehaviorSketchConfig = Field(description="Modified sketch configuration")
    updated_code: str = Field(description="Modified Python code")
    changes_made: List[str] = Field(description="Summary of changes applied")
    explanation: str = Field(description="Explanation of the modifications")


class SketchTemplates:
    """Pre-defined templates for common multi-behavior combinations. These are basically helpful for testing out and making sure the agent is working in the user's env."""
    
    @staticmethod
    def organic_flocking_with_trails() -> MultiBehaviorSketchConfig:
        """Flocking behavior combined with slime-like trails."""
        return MultiBehaviorSketchConfig(
            sketch_name="organic_flock_trails",
            description="Flocking particles that leave organic slime-like trails",
            behaviors=[
                BehaviorInstance(
                    behavior_type=BehaviorType.FLOCK,
                    particle_count=800,
                    species_indices=[0, 1],
                    flock_config=FlockBehaviorConfig(
                        separation_distance=0.15,
                        alignment_strength=0.7,
                        cohesion_force=0.5
                    ),
                    particle_shape=ParticleShape.CIRCLE
                ),
                BehaviorInstance(
                    behavior_type=BehaviorType.SLIME,
                    particle_count=400,
                    species_indices=[2],
                    slime_config=SlimeBehaviorConfig(
                        sense_angle=0.4,
                        sense_dist=15.0,
                        move_step=0.8
                    ),
                    particle_shape=ParticleShape.POINT
                )
            ],
            color_palette=ColorPalette(
                primary=(0.2, 0.8, 0.3),
                secondary=(0.8, 0.3, 0.2),
                accent=(0.9, 0.9, 0.1)
            )
        )
    
    @staticmethod
    def chaotic_multi_behavior() -> MultiBehaviorSketchConfig:
        return MultiBehaviorSketchConfig(
            sketch_name="chaotic_composition",
            description="Complex multi-behavior system with flocking, slime, and particle life",
            behaviors=[
                BehaviorInstance(
                    behavior_type=BehaviorType.FLOCK,
                    particle_count=600,
                    species_indices=[0],
                    flock_config=FlockBehaviorConfig(
                        separation_distance=0.2,
                        alignment_strength=0.9,
                        cohesion_force=0.4
                    )
                ),
                BehaviorInstance(
                    behavior_type=BehaviorType.SLIME,
                    particle_count=300,
                    species_indices=[1],
                    slime_config=SlimeBehaviorConfig(
                        sense_angle=0.6,
                        sense_dist=25.0
                    )
                ),
                BehaviorInstance(
                    behavior_type=BehaviorType.PARTICLE_LIFE,
                    particle_count=400,
                    species_indices=[2],
                    particle_life_config=ParticleLifeConfig(
                        attraction_radius=80.0,
                        repulsion_radius=30.0
                    )
                )
            ],
            color_palette=ColorPalette(
                primary=(0.9, 0.1, 0.1),
                secondary=(0.1, 0.9, 0.1),
                accent=(0.1, 0.1, 0.9)
            ),
            render_config=RenderConfig(
                diffuse_rate=0.95, 
                brightness=1.2
            )
        )