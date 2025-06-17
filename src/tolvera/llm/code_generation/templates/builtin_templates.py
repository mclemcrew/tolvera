"""
Templates for prompts 
"""

from typing import Dict

def get_builtin_templates() -> Dict[str, str]:
    return {
        'main_sketch': MAIN_SKETCH_TEMPLATE,
        'slime_flock_optimized': SLIME_FLOCK_OPTIMIZED_TEMPLATE
    }

# TODO: This is only used for the test_agent and can be deleted
def get_template_info() -> Dict[str, Dict[str, str]]:
    return {
        'main_sketch': {
            'description': 'General-purpose template for any multi-behavior sketch',
            'best_for': 'Any combination of behaviors, flexible and extensible',
            'behaviors': 'All supported (flock, slime, particle_life)'
        },
        'slime_flock_optimized': {
            'description': 'Specialized template optimized for slime + flock combinations',
            'best_for': 'Creatures that paint trails as they move',
            'behaviors': 'Flock + Slime (optimized order and parameters)'
        }
    }

# ==============================================================================
# MAIN SKETCH TEMPLATE
# ==============================================================================

MAIN_SKETCH_TEMPLATE = """{{ generate_imports(config) }}

def main(**kwargs):
{{ generate_docstring(config) }}
    
    # Create Tölvera instance with optimized settings
    tv = Tolvera({{ get_tolvera_kwargs(config) }}, **kwargs)
    
{{ optimize_for_slime_flock(config) }}
    
    # Configure all behaviors
    {% for behavior in config.behaviors %}
{{ generate_behavior_kernel(behavior, loop.index0) }}
    
    {% if not is_force_function(behavior) %}
    configure_{{ behavior.behavior_type.value }}_{{ loop.index0 }}()
    {% endif %}
    {% endfor %}
    
    @tv.render
    def _():
        # Background processing
        {{ generate_background_code(config.render_config) }}
        
        # Apply all behaviors in sequence
        {% for behavior in config.behaviors %}
        {{ generate_behavior_call(behavior) }}
        {% endfor %}
        
        # Render particles with species-specific shapes
        {% for behavior in config.behaviors %}
        {% if behavior.species_indices %}
        # Render {{ behavior.behavior_type.value }} particles (species {{ behavior.species_indices | species_list }})
        tv.px.particles(tv.p, tv.s.species({{ behavior.species_indices[0] }}), "{{ behavior.particle_shape.value }}")
        {% endif %}
        {% endfor %}
        
        return tv.px

if __name__ == '__main__':
    run(main)
"""

# ==============================================================================
# SLIME & FLOCK TEMPLATE TEST - This shouldn't be here, but I wanted to test this combination a bit more
# ==============================================================================

SLIME_FLOCK_OPTIMIZED_TEMPLATE = """{{ generate_imports(config) }}

def main(**kwargs):
{{ generate_docstring(config) }}
    
    # Tölvera instance optimized for slime + flock interaction
    tv = Tolvera({{ get_tolvera_kwargs(config) }}, osc=False, **kwargs)
    
    # === SLIME + FLOCK OPTIMIZATION ===
    # This sketch creates "creatures painting their environment" effects:
    # - Flocking behavior creates organized, natural movement patterns
    # - Slime behavior captures trails and paths for organic textures
    # - Together they create beautiful emergent ecosystem dynamics
    
    {% for behavior in config.behaviors %}
    {% if behavior.behavior_type.value == 'flock' %}
    @ti.kernel
    def configure_flocking():
        \"\"\"Configure flocking for natural creature movement.\"\"\"
        {% for species in behavior.species_indices %}
        # Flock species {{ species }} - balanced for trail interaction
        tv.s.flock_s[{{ species }}, {{ species }}].separate = {{ behavior.flock_config.separate }}
        tv.s.flock_s[{{ species }}, {{ species }}].align = {{ behavior.flock_config.align }}
        tv.s.flock_s[{{ species }}, {{ species }}].cohere = {{ behavior.flock_config.cohere }}
        tv.s.flock_s[{{ species }}, {{ species }}].radius = {{ behavior.flock_config.radius }}
        {% endfor %}
    
    {% elif behavior.behavior_type.value == 'slime' %}
    @ti.kernel
    def configure_slime_trails():
        \"\"\"Configure slime for capturing creature trails.\"\"\"
        {% for species in behavior.species_indices %}
        # Slime species {{ species }} - optimized for trail capture
        tv.s.slime_s[{{ species }}].sense_angle = {{ behavior.slime_config.sense_angle }}
        tv.s.slime_s[{{ species }}].sense_dist = {{ behavior.slime_config.sense_dist }}
        tv.s.slime_s[{{ species }}].move_angle = {{ behavior.slime_config.move_angle }}
        tv.s.slime_s[{{ species }}].move_dist = {{ behavior.slime_config.move_dist }}
        tv.s.slime_s[{{ species }}].evaporate = {{ behavior.slime_config.evaporate }}
        {% endfor %}
    
    {% endif %}
    {% endfor %}
    
    # Initialize all behaviors
    {% for behavior in config.behaviors %}
    {% if behavior.behavior_type.value == 'flock' %}
    configure_flocking()
    {% elif behavior.behavior_type.value == 'slime' %}
    configure_slime_trails()
    {% endif %}
    {% endfor %}
    
    @tv.render
    def _():
        # Diffusion creates persistent but fading trails
        {{ generate_background_code(config.render_config) }}
        
        # Apply behaviors in optimal order for slime + flock
        {% for behavior in config.behaviors %}
        {% if behavior.behavior_type.value == 'flock' %}
        tv.v.flock(tv.p)  # Creatures move in flocking patterns
        {% elif behavior.behavior_type.value == 'slime' %}
        tv.v.slime(tv.p, tv.s.species())  # Trails follow and enhance movement
        {% endif %}
        {% endfor %}
        
        # Render with optimized shapes for performance
        {% for behavior in config.behaviors %}
        {% if behavior.species_indices %}
        {% if behavior.behavior_type.value == 'flock' %}
        # Flock particles - visible creatures
        tv.px.particles(tv.p, tv.s.species({{ behavior.species_indices[0] }}), "circle")
        {% elif behavior.behavior_type.value == 'slime' %}
        # Slime particles - efficient trail points
        tv.px.particles(tv.p, tv.s.species({{ behavior.species_indices[0] }}), "point")
        {% endif %}
        {% endif %}
        {% endfor %}
        
        return tv.px

if __name__ == '__main__':
    run(main)
"""

# ==============================================================================
# FUTURE TEMPLATE PLACEHOLDERS
# ==============================================================================