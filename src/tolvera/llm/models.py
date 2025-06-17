"""
Pydantic Models based on the Tölvera library
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Tuple
from enum import Enum

class ParticleShape(str, Enum):
    CIRCLE = "circle"
    POINT = "point"
    RECT = "rect"


class BackgroundBehavior(str, Enum):
    DIFFUSE = "diffuse"
    CLEAR = "clear"
    FADE = "fade"


class BehaviorType(str, Enum):
    # This isn't exactly from the forces.py file, but I can try to add different ones to these too.
    FLOCK = "flock"
    SLIME = "slime"
    PARTICLE_LIFE = "particle_life"
    ATTRACT = "attract"
    REPEL = "repel"
    GRAVITATE = "gravitate"
    NOISE = "noise"
    CENTRIPETAL = "centripetal"


class ColorPalette(BaseModel):
    background: Tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0),
        description="RGB background color (0.0-1.0)"
    )
    primary: Tuple[float, float, float] = Field(
        description="Primary particle color (0.0-1.0)"
    )
    
    # This is primary used just for the configuration file but is to TODO because it only works for two species
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
            # TODO: I have an error happening here and I'm not sure what's going wrong with values outside of 1.  Need to look into the library a bit more.
            raise ValueError("Values must be between 0.0 and 1.0")
        return v


class FlockBehaviorConfig(BaseModel):
    separate: float = Field(default=0.3, ge=0.01, le=1.0)
    align: float = Field(default=0.7, ge=0.01, le=1.0)
    cohere: float = Field(default=0.5, ge=0.01, le=1.0)
    radius: float = Field(default=0.2, ge=0.01, le=1.0)


class SlimeBehaviorConfig(BaseModel):
    sense_angle: float = Field(default=0.5, ge=0.0, le=1.0)
    sense_dist: float = Field(default=0.4, ge=0.0, le=1.0)
    move_angle: float = Field(default=0.3, ge=0.0, le=1.0)
    move_dist: float = Field(default=0.25, ge=0.0, le=1.0)
    evaporate: float = Field(default=0.99, ge=0.0, le=1.0)


# TODO: look back at library for these
class ParticleLifeConfig(BaseModel):
    attract: float = Field(default=0.1, ge=-0.5, le=0.5)
    radius: float = Field(default=150.0, ge=100.0, le=300.0)


class AttractForceConfig(BaseModel):
    pos: Tuple[float, float] = Field(default=(0.5, 0.5))
    mass: float = Field(default=0.5, ge=0.0, le=2.0)
    radius: float = Field(default=200.0, ge=50.0, le=500.0)


class RepelForceConfig(BaseModel):
    pos: Tuple[float, float] = Field(default=(0.5, 0.5))
    mass: float = Field(default=0.5, ge=0.0, le=2.0)
    radius: float = Field(default=200.0, ge=50.0, le=500.0)


class GravitateForceConfig(BaseModel):
    G: float = Field(default=0.1, ge=0.0, le=1.0)
    radius: float = Field(default=300.0, ge=100.0, le=600.0)


class NoiseForceConfig(BaseModel):
    weight: float = Field(default=0.1, ge=0.0, le=1.0)


class CentripetalForceConfig(BaseModel):
    centre: Tuple[float, float] = Field(default=(0.5, 0.5))
    direction: int = Field(default=1, ge=0, le=1)
    weight: float = Field(default=0.5, ge=0.0, le=2.0)


class BehaviorInstance(BaseModel):
    behavior_type: BehaviorType = Field(description="Type of behavior")
    
    particle_count: int = Field(
        default=512, ge=0, le=5000,
        description="Number of particles for this behavior (can be 0 for global forces)"
    )
    
    species_indices: List[int] = Field(default_factory=lambda: [0])
    flock_config: Optional[FlockBehaviorConfig] = None
    slime_config: Optional[SlimeBehaviorConfig] = None
    particle_life_config: Optional[ParticleLifeConfig] = None
    attract_config: Optional[AttractForceConfig] = None
    repel_config: Optional[RepelForceConfig] = None
    gravitate_config: Optional[GravitateForceConfig] = None
    noise_config: Optional[NoiseForceConfig] = None
    centripetal_config: Optional[CentripetalForceConfig] = None
    particle_shape: ParticleShape = Field(default=ParticleShape.CIRCLE)
    color: Optional[Tuple[float, float, float]] = None

    @model_validator(mode='after')
    def validate_behavior_config(self):
        """Verify the correct config is set based on behavior type."""
        # This validator is here so that that if a behavior is selected by the LLM, its corresponding config object is created, even if it's empty.
        config_map = {
            BehaviorType.FLOCK: ('flock_config', FlockBehaviorConfig),
            BehaviorType.SLIME: ('slime_config', SlimeBehaviorConfig),
            BehaviorType.PARTICLE_LIFE: ('particle_life_config', ParticleLifeConfig),
            BehaviorType.ATTRACT: ('attract_config', AttractForceConfig),
            BehaviorType.REPEL: ('repel_config', RepelForceConfig),
            BehaviorType.GRAVITATE: ('gravitate_config', GravitateForceConfig),
            BehaviorType.NOISE: ('noise_config', NoiseForceConfig),
            BehaviorType.CENTRIPETAL: ('centripetal_config', CentripetalForceConfig),
        }
        
        if self.behavior_type in config_map:
            config_name, config_class = config_map[self.behavior_type]
            if getattr(self, config_name) is None:
                setattr(self, config_name, config_class())
        
        return self

    @model_validator(mode='after')
    def check_particle_count_for_non_forces(self):
        # TODO a lot of the time the model will choose something like 5000 particles.  Can't handle that.  We need to fix this.
        """
        Enforce particle_count >= 10 for behaviors that have particles, but allow it to be 0 for global forces.
        """
        is_force_behavior = self.behavior_type in [
            BehaviorType.ATTRACT,
            BehaviorType.REPEL,
            BehaviorType.GRAVITATE,
            BehaviorType.NOISE,
            BehaviorType.CENTRIPETAL
        ]

        if not is_force_behavior and self.particle_count < 10:
            raise ValueError(f"'{self.behavior_type.value}' behaviors must have at least 10 particles.")
            
        return self


class RenderConfig(BaseModel):
    """Global rendering config for the entire sketch."""
    background_behavior: BackgroundBehavior = Field(default=BackgroundBehavior.DIFFUSE)
    diffuse_rate: float = Field(default=0.99, ge=0.0, le=1.0)
    window_width: int = Field(default=1920, ge=400, le=3840)
    window_height: int = Field(default=1080, ge=300, le=2160)


class MultiBehaviorSketchConfig(BaseModel):
    # TODO need to test this with more than 2 configurations combined.  It's more combinatorial right now.
    """
    Configuration for Tölvera sketches that combine multiple behaviors.
    """
    sketch_name: str = Field(min_length=1, max_length=100)
    description: str
    behaviors: List[BehaviorInstance] = Field(min_length=1)
    render_config: RenderConfig = Field(default_factory=RenderConfig)
    color_palette: ColorPalette
    global_speed: float = Field(default=1.0, ge=0.1, le=5.0)
    substeps: int = Field(default=1, ge=1, le=10)

    @field_validator('sketch_name')
    @classmethod
    def validate_sketch_name(cls, v: str) -> str:
        import re
        safe_name = re.sub(r'[^\w\-_]', '_', v)
        if not safe_name:
            raise ValueError("Sketch name cannot be empty after cleaning")
        return safe_name

    def get_total_particle_count(self) -> int:
        return sum(b.particle_count for b in self.behaviors)
    
    def get_max_species_index(self) -> int:
        if not self.behaviors: return 0
        all_indices = [idx for b in self.behaviors for idx in b.species_indices]
        return max(all_indices) if all_indices else 0

    def get_behavior_summary(self) -> str:
        behavior_types = [b.behavior_type.value for b in self.behaviors]
        return f"Multi-behavior sketch with: {', '.join(behavior_types)}"


class SketchGenerationResponse(BaseModel):
    config: MultiBehaviorSketchConfig
    python_code: str
    explanation: str
    suggestions: List[str] = Field(default_factory=list)


class SketchModificationResponse(BaseModel):
    updated_config: MultiBehaviorSketchConfig
    updated_code: str
    changes_made: List[str]
    explanation: str


# This is just for fallback template.  Not used, but tested in the test script to make sure it works.  But again, this is a fallback here.
class SketchTemplates:
    @staticmethod
    def organic_flocking_with_trails() -> MultiBehaviorSketchConfig:
        return MultiBehaviorSketchConfig(
            sketch_name="organic_flock_trails",
            description="Flocking particles that leave organic slime-like trails",
            behaviors=[
                BehaviorInstance(behavior_type=BehaviorType.FLOCK, particle_count=800, species_indices=[0, 1], flock_config=FlockBehaviorConfig(separate=0.25, align=0.7, cohere=0.5, radius=0.15), particle_shape=ParticleShape.CIRCLE),
                BehaviorInstance(behavior_type=BehaviorType.SLIME, particle_count=400, species_indices=[2], slime_config=SlimeBehaviorConfig(sense_angle=0.4, sense_dist=0.3, move_dist=0.2), particle_shape=ParticleShape.POINT)
            ],
            color_palette=ColorPalette(primary=(0.2, 0.8, 0.3), secondary=(0.8, 0.3, 0.2), accent=(0.9, 0.9, 0.1))
        )
    
    @staticmethod
    def chaotic_multi_behavior() -> MultiBehaviorSketchConfig:
        return MultiBehaviorSketchConfig(
            sketch_name="chaotic_composition",
            description="Complex multi-behavior system with flocking, slime, and particle life",
            behaviors=[
                BehaviorInstance(behavior_type=BehaviorType.FLOCK, particle_count=600, species_indices=[0], flock_config=FlockBehaviorConfig(separate=0.4, align=0.9, cohere=0.3, radius=0.25)),
                BehaviorInstance(behavior_type=BehaviorType.SLIME, particle_count=300, species_indices=[1], slime_config=SlimeBehaviorConfig(sense_angle=0.6, sense_dist=0.5, move_dist=0.3)),
                BehaviorInstance(behavior_type=BehaviorType.PARTICLE_LIFE, particle_count=400, species_indices=[2], particle_life_config=ParticleLifeConfig(attract=0.2, radius=180.0))
            ],
            color_palette=ColorPalette(primary=(0.9, 0.1, 0.1), secondary=(0.1, 0.9, 0.1), accent=(0.1, 0.1, 0.9)),
            render_config=RenderConfig(diffuse_rate=0.95, brightness=1.2)
        )