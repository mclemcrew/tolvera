"""
Generates the code from the jinja templates.
"""

import logging
from typing import Dict, Optional

from jinja2 import Environment, BaseLoader
from ..models import MultiBehaviorSketchConfig, BehaviorType, BackgroundBehavior
from .templates import get_builtin_templates

logger = logging.getLogger(__name__)


class StringTemplateLoader(BaseLoader):

    def __init__(self, templates: Dict[str, str]):
        self.templates = templates

    def get_source(self, environment, template):
        if template not in self.templates:
            available = ", ".join(self.templates.keys())
            # Just checking that we have the template path when testing.  TODO: Can delete from here.
            raise Exception(f"Template '{template}' not found. Available: {available}")

        source = self.templates[template]
        return source, None, lambda: True


class JinjaCodeGenerator:
    def __init__(self, template_dir: Optional[str] = None):
        """
        Initialize the enhanced code generator.

        Args:
            template_dir: Optional external template directory if we want to have an external template form that grows over time.
        """
        self.template_dir = template_dir
        self._setup_jinja_environment()
        logger.info(f"Code generator initialized with {'external' if template_dir else 'built-in'} templates")

    def _setup_jinja_environment(self):
        templates = get_builtin_templates()
        loader = StringTemplateLoader(templates)

        self.env = Environment(
            loader=loader,
            trim_blocks=True,
            lstrip_blocks=True,
            line_statement_prefix='##'
        )

        self.env.globals.update({
            'generate_behavior_kernel': self._generate_behavior_kernel,
            'generate_behavior_call': self._generate_behavior_call,
            'generate_background_code': self._generate_background_code,
            # Color is still a problem for me.
            'format_color': self._format_color,
            'get_tolvera_kwargs': self._get_tolvera_kwargs,
            'optimize_for_slime_flock': self._optimize_for_slime_flock,
            'generate_imports': self._generate_imports,
            'generate_docstring': self._generate_docstring,
            # This was added because I kept getting errors when adding force params (aka we didn't need a kernal)
            'is_force_function': self._is_force_function
        })

        self.env.filters.update({
            'behavior_name': lambda behavior: behavior.behavior_type.value,
            'safe_name': self._safe_name,
            'indent': self._indent_lines,
            'species_list': lambda indices: ', '.join(map(str, indices))
        })

    def _is_force_function(self, behavior) -> bool:
        return behavior.behavior_type in [BehaviorType.ATTRACT, BehaviorType.REPEL,
                                          BehaviorType.GRAVITATE, BehaviorType.NOISE,
                                          BehaviorType.CENTRIPETAL]

    def generate_python_code(self, config: MultiBehaviorSketchConfig) -> str:
        try:
            template = self.env.get_template('main_sketch')
            generated_code = template.render(config=config)
            self._validate_generated_code(generated_code, config)
            # Just want to make sure it's not just generating 1 line or something like that.
            logger.info(f"✅ Generated {len(generated_code.splitlines())} lines of code for '{config.sketch_name}'")
            return generated_code
        except Exception as e:
            logger.error(f"❌ Code generation failed: {e}")
            raise RuntimeError(f"Template rendering failed: {e}")

    # This is based on a lot of examples I found and when playing around with the library was catching this...just need to setup a kernal decorator before we get into the rest of the items.
    def _generate_behavior_kernel(self, behavior, index: int) -> str:
        kernel_generators = {
            BehaviorType.FLOCK: self._render_flock_kernel,
            BehaviorType.SLIME: self._render_slime_kernel,
            BehaviorType.PARTICLE_LIFE: self._render_particle_life_kernel
        }
        generator = kernel_generators.get(behavior.behavior_type)
        if generator:
            return generator(behavior, index)
        elif self._is_force_function(behavior):
            return f"    # {behavior.behavior_type.value} force - no configuration needed"
        else:
            return f"    # Unknown behavior type: {behavior.behavior_type}"


    # **************************************************************************************
    # TODO: Maybe we should rethink this idea of constantly defining the behavior explicitly
    # **************************************************************************************
    def _render_flock_kernel(self, behavior, index: int) -> str:
        lines = [f"    @ti.kernel", f"    def configure_flock_{index}():", f'        """Configure flocking behavior for species {behavior.species_indices}."""']
        for species_idx in behavior.species_indices:
            config = behavior.flock_config
            lines.extend([f"        # Flocking parameters for species {species_idx}", f"        tv.s.flock_s[{species_idx}, {species_idx}].separate = {config.separate}", f"        tv.s.flock_s[{species_idx}, {species_idx}].align = {config.align}", f"        tv.s.flock_s[{species_idx}, {species_idx}].cohere = {config.cohere}", f"        tv.s.flock_s[{species_idx}, {species_idx}].radius = {config.radius}"])
        return '\n'.join(lines)

    def _render_slime_kernel(self, behavior, index: int) -> str:
        lines = [f"    @ti.kernel", f"    def configure_slime_{index}():", f'        """Configure slime behavior for species {behavior.species_indices}."""']
        for species_idx in behavior.species_indices:
            config = behavior.slime_config
            lines.extend([f"        # Slime parameters for species {species_idx}", f"        tv.s.slime_s[{species_idx}].sense_angle = {config.sense_angle}", f"        tv.s.slime_s[{species_idx}].sense_dist = {config.sense_dist}", f"        tv.s.slime_s[{species_idx}].move_angle = {config.move_angle}", f"        tv.s.slime_s[{species_idx}].move_dist = {config.move_dist}", f"        tv.s.slime_s[{species_idx}].evaporate = {config.evaporate}"])
        return '\n'.join(lines)

    def _render_particle_life_kernel(self, behavior, index: int) -> str:
        lines = [f"    @ti.kernel", f"    def configure_particle_life_{index}():", f'        """Configure particle life behavior for species {behavior.species_indices}."""']
        for species_idx in behavior.species_indices:
            config = behavior.particle_life_config
            lines.extend([f"        # Particle life for species {species_idx}", f"        tv.s.plife[{species_idx}, {species_idx}].attract = {config.attract}", f"        tv.s.plife[{species_idx}, {species_idx}].radius = {config.radius}"])
        return '\n'.join(lines)

    # Again, mostly from forces.py, but we need to check on this. TODO: Need to restructure this.
    def _generate_behavior_call(self, behavior) -> str:
        behavior_calls = {
            BehaviorType.FLOCK: "tv.v.flock(tv.p)",
            BehaviorType.SLIME: "tv.v.slime(tv.p, tv.s.species())",
            BehaviorType.PARTICLE_LIFE: "tv.v.plife(tv.p)",
            BehaviorType.ATTRACT: self._generate_attract_call(behavior),
            BehaviorType.REPEL: self._generate_repel_call(behavior),
            BehaviorType.GRAVITATE: self._generate_gravitate_call(behavior),
            BehaviorType.NOISE: self._generate_noise_call(behavior),
            BehaviorType.CENTRIPETAL: self._generate_centripetal_call(behavior)
        }
        return behavior_calls.get(behavior.behavior_type, f"# Unknown behavior: {behavior.behavior_type}")

    def _generate_attract_call(self, behavior) -> str:
        config = behavior.attract_config
        if config is None: return "# Attract config missing"
        pos_x = config.pos[0] * 1920
        pos_y = config.pos[1] * 1080
        return f"tv.v.attract(tv.p, ti.Vector([{pos_x}, {pos_y}]), {config.mass}, {config.radius})"

    def _generate_repel_call(self, behavior) -> str:
        config = behavior.repel_config
        if config is None: return "# Repel config missing"
        pos_x = config.pos[0] * 1920
        pos_y = config.pos[1] * 1080
        return f"tv.v.repel(tv.p, ti.Vector([{pos_x}, {pos_y}]), {config.mass}, {config.radius})"

    def _generate_gravitate_call(self, behavior) -> str:
        config = behavior.gravitate_config
        if config is None: return "# Gravitate config missing"
        return f"tv.v.gravitate(tv.p, {config.G}, {config.radius})"

    def _generate_noise_call(self, behavior) -> str:
        config = behavior.noise_config
        if config is None: return "# Noise config missing"
        return f"tv.v.noise(tv.p, {config.weight})"

    def _generate_centripetal_call(self, behavior) -> str:
        config = behavior.centripetal_config
        if config is None: return "# Centripetal config missing"
        centre_x = config.centre[0] * 1920
        centre_y = config.centre[1] * 1080
        return f"tv.v.centripetal(tv.p, ti.Vector([{centre_x}, {centre_y}]), {config.direction}, {config.weight})"

    def _generate_background_code(self, render_config) -> str:
        if render_config.background_behavior == BackgroundBehavior.DIFFUSE:
            # I had some problems with brightness here.  I just removed it, but I think it's part of the pixels.py
            return f"tv.px.diffuse({render_config.diffuse_rate})"
        elif render_config.background_behavior == BackgroundBehavior.CLEAR:
            return "tv.px.clear()"
        else:
            return "# No background processing"
        

    def _format_color(self, color_tuple) -> str:
        if color_tuple and len(color_tuple) >= 3:
            r, g, b = color_tuple[:3]
            return f"[{r:.3f}, {g:.3f}, {b:.3f}, 1.0]"
        return "[1.0, 1.0, 1.0, 1.0]"

    def _get_tolvera_kwargs(self, config) -> str:
        kwargs = {'n': config.get_total_particle_count(), 'species': config.get_max_species_index() + 1}
        if self._is_slime_flock_combination(config):
            kwargs['osc'] = False
        return ", ".join([f"{k}={v}" for k, v in kwargs.items()])

    def _optimize_for_slime_flock(self, config) -> str:
        if self._is_slime_flock_combination(config):
            return """    # Optimized for slime + flock interaction:
                          # - Flocking creates organized movement patterns
                          # - Slime captures trails for organic texture effects
                          # - Performance balanced for smooth real-time interaction"""
        return ""

    def _generate_imports(self, config) -> str:
        imports = ["from tolvera import Tolvera, run", "import taichi as ti"]
        if len(config.behaviors) > 2:
            imports.append("import numpy as np  # For complex multi-behavior interactions")
        return '\n'.join(imports)

    def _generate_docstring(self, config) -> str:
        behavior_summary = config.get_behavior_summary()
        total_particles = config.get_total_particle_count()
        docstring = f'    """\n    {config.sketch_name}\n    \n    {config.description}\n    \n'
        docstring += f'    Behaviors: {behavior_summary}\n'
        docstring += f'    Total particles: {total_particles}\n'
        if self._is_slime_flock_combination(config):
            docstring += '    \n    Features slime + flock interaction for organic trail effects.\n'
        docstring += '    """'
        return docstring

    def _is_slime_flock_combination(self, config) -> bool:
        behavior_types = [b.behavior_type for b in config.behaviors]
        return BehaviorType.FLOCK in behavior_types and BehaviorType.SLIME in behavior_types

    def _has_slime_behavior_in_config(self) -> bool:
        return False

    def _safe_name(self, name: str) -> str:
        import re
        return re.sub(r'[^\w\-_]', '_', name.lower())

    def _indent_lines(self, text: str, spaces: int = 4) -> str:
        lines = text.split('\n')
        indented = [' ' * spaces + line if line.strip() else line for line in lines]
        return '\n'.join(indented)

    def _validate_generated_code(self, code: str, config: MultiBehaviorSketchConfig):
        required_patterns = ["from tolvera import", "@ti.kernel", "def main(", "@tv.render"]
        for pattern in required_patterns:
            if pattern not in code:
                raise ValueError(f"Generated code missing required pattern: {pattern}")
        for behavior in config.behaviors:
            if behavior.behavior_type == BehaviorType.FLOCK and "tv.v.flock" not in code:
                raise ValueError("Flock behavior configured but not called in render loop")
            elif behavior.behavior_type == BehaviorType.SLIME and "tv.v.slime" not in code:
                raise ValueError("Slime behavior configured but not called in render loop")

def create_code_generator(template_dir: Optional[str] = None) -> JinjaCodeGenerator:
    return JinjaCodeGenerator(template_dir)