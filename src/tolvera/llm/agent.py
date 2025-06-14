"""
Tölvera PydanticAI Agent
"""


import requests
from typing import List, Dict, Any, Tuple
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel

try:
    from .models import (
        MultiBehaviorSketchConfig, 
        SketchGenerationResponse,
        SketchModificationResponse,
        BehaviorInstance,
        BehaviorType,
        FlockBehaviorConfig,
        SlimeBehaviorConfig,
        ParticleLifeConfig,
        ColorPalette,
        RenderConfig,
        ParticleShape,
        BackgroundBehavior,
        SketchTemplates
    )
except ImportError:
    # I was just having issues importing things here.
    from models import (
        MultiBehaviorSketchConfig, 
        SketchGenerationResponse,
        SketchModificationResponse,
        BehaviorInstance,
        BehaviorType,
        FlockBehaviorConfig,
        SlimeBehaviorConfig,
        ParticleLifeConfig,
        ColorPalette,
        RenderConfig,
        ParticleShape,
        BackgroundBehavior,
        SketchTemplates
    )

class TolveraSketchAgent:
    """
    PydanticAI agent for generating multi-behavior Tölvera sketches.
    """
    
    def __init__(self, 
                 model_name: str = "qwen2.5:14b",
                 ollama_host: str = "http://localhost:11434"):
        """
        Initialize the Tölvera sketch generation agent.
        
        Args:
            model_name: Ollama model to use (qwen2.5:14b because it works with tools, but need to test more)
            ollama_host: Ollama server URL
        """
        self.model_name = model_name
        self.ollama_host = ollama_host
        
        self._validate_ollama_setup()
        
        from pydantic_ai.providers.openai import OpenAIProvider
        
        self.model = OpenAIModel(
            model_name=model_name,  
            provider=OpenAIProvider(
                base_url=f"{ollama_host}/v1",
                api_key="ollama" 
            )
        )
        
        self.generation_agent = Agent(
            model=self.model,
            result_type=SketchGenerationResponse,
            system_prompt=self._build_generation_system_prompt(),
            tools=[
                self._validate_behavior_combination,
                self._suggest_performance_optimizations,
                self._generate_color_palette
            ]
        )
        
        self.modification_agent = Agent(
            model=self.model,
            result_type=SketchModificationResponse,
            system_prompt=self._build_modification_system_prompt(),
            tools=[
                self._analyze_existing_sketch,
                self._suggest_modifications
            ]
        )
    
    def _validate_ollama_setup(self):
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=10)
            if response.status_code != 200:
                raise ConnectionError(f"Ollama not responding at {self.ollama_host}")
            
            models = response.json().get('models', [])
            available_models = [m['name'] for m in models]
            
            print(f"Available Ollama models: {available_models}")
            
            # Models known to support tools (function calling) - prioritize these
            tool_compatible_models = [
                "qwen2.5:latest", "qwen2.5:14b", "qwen2.5:7b", "qwen2.5:3b",
                "llama3.1:latest", "llama3.1:8b", "llama3.1:70b", 
                "gemma3:latest", "gemma3:2b", "gemma3:9b",
                "qwen3:4b", "qwen3:latest",
                "llama3.2:latest", "llama3.2:3b", "llama3.2:1b",
                "codestral:latest", "mixtral:latest"
            ]
            
            # Models that don't support tools (avoid these for PydanticAI agents)
            non_tool_models = [
                "deepseek-r1:8b", "deepseek-r1:latest",  # DeepSeek R1 series
            ]
            
            # Check if requested model is available and tool-compatible
            if self.model_name in available_models:
                if self.model_name in non_tool_models:
                    print(f"⚠️  Model '{self.model_name}' doesn't support tools")
                    print("🔄 Looking for tool-compatible alternative...")
                else:
                    print(f"✅ Requested model '{self.model_name}' is available and should support tools")
                    return  # Use the requested model
            else:
                print(f"⚠️  Model '{self.model_name}' not found")
            
            best_model = None
            
            # This is arbitary 
            preferred_order = [
                "qwen2.5:latest",
                "gemma3:latest",    
                "llama3.1:latest",  
                "qwen3:4b",         
                "llama3.2:latest",  
            ]
            
            for preferred in preferred_order:
                if preferred in available_models:
                    best_model = preferred
                    break
            
            ## Fallback on this
            if not best_model:
                for model in available_models:
                    if any(compatible in model for compatible in tool_compatible_models):
                        if model not in non_tool_models:
                            best_model = model
                            break
            
            if best_model:
                self.model_name = best_model
                print(f"✅ Using tool-compatible model: {self.model_name}")
            else:
                print("❌ No tool-compatible models found")
                print("Install a recommended model that supports tools:")
                print("   ollama pull qwen2.5:latest")
                print("   ollama pull gemma3:latest") 
                print("   ollama pull llama3.1:latest")
                raise RuntimeError("No tool-compatible models available")
            
            # Test OpenAI-compatible endpoint
            try:
                test_response = requests.get(f"{self.ollama_host}/v1/models", timeout=5)
                if test_response.status_code == 200:
                    print("✅ Ollama OpenAI-compatible endpoint is working")
                else:
                    print("⚠️  Ollama OpenAI endpoint returned non-200 status")
            except requests.RequestException:
                print("⚠️  Could not test OpenAI-compatible endpoint")
                
        except requests.RequestException as e:
            raise ConnectionError(f"Cannot connect to Ollama at {self.ollama_host}: {e}")
    
    def _build_generation_system_prompt(self) -> str:
        """Build the system prompt for sketch generation."""
        return """You are an expert Tölvera artificial life artist and programmer. You create beautiful, emergent multi-behavior compositions using particle systems.

CRITICAL: You can now combine MULTIPLE behaviors in a single sketch for complex artistic compositions.

## Supported Multi-Behavior Patterns

You can create sketches that combine:
- Flocking + Slime: Birds that leave organic trails
- Particle Life + Flocking: Self-organizing systems with group dynamics  
- All Three: Complex ecosystems with multiple interaction types
- Layered Behaviors: Different species with different behaviors

## Tölvera API Requirements (CRITICAL)

ALL generated code must follow these EXACT patterns:

```python
from tolvera import Tolvera, run
import taichi as ti  # Required for @ti.kernel

def main(**kwargs):
    tv = Tolvera(**kwargs)
    
    # CRITICAL: Use @ti.kernel for ALL state assignments
    @ti.kernel  
    def configure_flock_behavior():
        # Configure flocking for species 0
        tv.s.flock_s[0, 0].separate = 0.1
        tv.s.flock_s[0, 0].align = 0.8
        tv.s.flock_s[0, 0].cohere = 0.6
    
    @ti.kernel
    def configure_slime_behavior():
        # Configure slime for species 1
        tv.s.slime_s[1].sense_angle = 0.5
        tv.s.slime_s[1].sense_dist = 20.0
        tv.s.slime_s[1].move_angle = 0.3
        tv.s.slime_s[1].move_step = 1.0
    
    # Call configuration kernels
    configure_flock_behavior()
    configure_slime_behavior()
    
    @tv.render
    def _():
        # Background handling
        tv.px.diffuse(0.99)
        
        # Apply behaviors to appropriate species
        tv.v.flock(tv.p)  # Affects all species with flock config
        tv.v.slime(tv.p, tv.s.species())  # Slime behavior
        
        # Render particles
        tv.px.particles(tv.p, tv.s.species(), "circle")
        
        return tv.px

if __name__ == '__main__':
    run(main)
```

## Multi-Behavior Guidelines

1. Species Assignment: Assign different species to different behaviors
2. Parameter Harmony: Make sure behavior parameters work well together
3. Performance: Total particles should stay under 5000 for good performance
4. Visual Balance: Consider how behaviors will interact visually
5. Kernel Usage: ALL tv.s.* assignments MUST be in @ti.kernel functions

## Style Interpretation

- "organic" → Combine flocking + slime for natural movement
- "chaotic" → Multiple behaviors with conflicting parameters  
- "flowing" → High diffusion, gentle slime parameters
- "energetic" → Fast movement, low diffusion, sharp interactions
- "minimal" → Single behavior, clean parameters
- "complex" → All three behaviors, intricate interactions

## Example Multi-Behavior Configs

Organic Flocking with Trails:
- Species 0-1: Flocking (birds)
- Species 2: Slime (trails they leave)
- 800 flock + 400 slime particles

Ecosystem Simulation:  
- Species 0: Flocking (prey)
- Species 1: Particle Life (predator)
- Species 2: Slime (environment)

Generate configurations that create beautiful, emergent artistic compositions!"""

    def _build_modification_system_prompt(self) -> str:
        """Build the system prompt for sketch modification."""
        return """You are an expert at modifying existing Tölvera multi-behavior sketches.

Your job is to take an existing sketch configuration and modify it based on user requests while maintaining the artistic integrity and technical correctness.

## Modification Strategies

1. Behavior Addition: Add new behaviors to existing sketches
2. Parameter Tuning: Adjust existing behavior parameters
3. Species Rebalancing: Change which species use which behaviors
4. Visual Enhancement: Modify colors, shapes, rendering
5. Performance Optimization: Reduce particle counts, optimize parameters

## Modification Guidelines

- Preserve the core artistic concept unless explicitly asked to change it
- Maintain technical correctness (proper species indices, kernel patterns)
- Keep total particle count reasonable (<5000)
- Explain what each change accomplishes artistically
- Suggest complementary changes when appropriate

Always generate complete, working configurations and code."""

    # Tool functions for the agents
    async def _validate_behavior_combination(
        self, 
        ctx: RunContext,
        behaviors: List[str],
        particle_counts: List[int]
    ) -> Dict[str, Any]:
        """Tool to validate if behavior combinations work well together."""
        total_particles = sum(particle_counts)
        
        validation = {
            "is_valid": True,
            "warnings": [],
            "suggestions": []
        }
        
        if total_particles > 5000:
            validation["warnings"].append(f"High particle count ({total_particles}) may impact performance")
        
        if "flock" in behaviors and "slime" in behaviors:
            validation["suggestions"].append("Flocking + slime creates beautiful organic trail effects")
        
        if len(behaviors) > 2:
            validation["suggestions"].append("Complex multi-behavior systems create rich emergent patterns")
        
        return validation

    async def _suggest_performance_optimizations(
        self,
        ctx: RunContext,
        total_particles: int,
        behavior_count: int
    ) -> List[str]:
        """Tool to suggest performance optimizations."""
        suggestions = []
        
        if total_particles > 3000:
            suggestions.append("Consider reducing particle counts for better performance")
        
        if behavior_count > 2:
            suggestions.append("Multiple behaviors may impact frame rate - test on target hardware")
        
        suggestions.append("Use 'point' particles for slime behaviors for better performance")
        suggestions.append("Keep diffusion rate between 0.95-0.99 for good trails without performance impact")
        
        return suggestions

    async def _generate_color_palette(
        self,
        ctx: RunContext,
        mood: str,
        behavior_types: List[str]
    ) -> Dict[str, Tuple[float, float, float]]:
        """Tool to generate appropriate color palettes."""
        palettes = {
            "organic": {
                "primary": (0.2, 0.8, 0.3),
                "secondary": (0.6, 0.9, 0.4),
                "accent": (0.1, 0.6, 0.2)
            },
            "energetic": {
                "primary": (1.0, 0.3, 0.1),
                "secondary": (1.0, 0.8, 0.0),
                "accent": (0.9, 0.1, 0.4)
            },
            "dreamy": {
                "primary": (0.6, 0.4, 0.9),
                "secondary": (0.8, 0.6, 1.0),
                "accent": (0.4, 0.8, 0.9)
            },
            "minimal": {
                "primary": (0.9, 0.9, 0.9),
                "secondary": (0.7, 0.7, 0.7),
                "accent": (0.5, 0.5, 0.5)
            }
        }
        
        return palettes.get(mood, palettes["organic"])

    async def _analyze_existing_sketch(
        self,
        ctx: RunContext,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Tool to analyze existing sketch configurations."""
        analysis = {
            "behavior_count": len(config.get("behaviors", [])),
            "total_particles": sum(b.get("particle_count", 0) for b in config.get("behaviors", [])),
            "species_used": [],
            "complexity": "low"
        }
        
        # Extract species indices
        for behavior in config.get("behaviors", []):
            analysis["species_used"].extend(behavior.get("species_indices", []))
        
        # Determine complexity by number of species
        if analysis["behavior_count"] > 2:
            analysis["complexity"] = "high"
        elif analysis["behavior_count"] == 2:
            analysis["complexity"] = "medium"
        
        return analysis

    async def _suggest_modifications(
        self,
        ctx: RunContext,
        current_analysis: Dict[str, Any],
        modification_request: str
    ) -> List[str]:
        """Tool to suggest specific modifications."""
        suggestions = []
        
        if "more particles" in modification_request.lower():
            suggestions.append("Increase particle counts for existing behaviors")
        
        if "add" in modification_request.lower() and "behavior" in modification_request.lower():
            suggestions.append("Add a new behavior instance with appropriate species assignment")
        
        if "organic" in modification_request.lower():
            suggestions.append("Adjust parameters for more natural, flowing movement")
        
        if "chaotic" in modification_request.lower():
            suggestions.append("Increase randomness and reduce alignment/cohesion")
        
        return suggestions

    async def generate_sketch(self, description: str, **kwargs) -> SketchGenerationResponse:
        """
        Generate a new multi-behavior Tölvera sketch from natural language.
        
        Args:
            description: Natural language description of desired artwork
            **kwargs: Additional parameters
            
        Returns:
            SketchGenerationResponse with configuration and code
        """
        print(f"🎨 Generating sketch: {description}")
        
        try:
            result = await self.generation_agent.run(
                f"""Create a Tölvera multi-behavior sketch based on this description: "{description}"

Generate a complete MultiBehaviorSketchConfig with:
1. Multiple behavior instances that work well together artistically
2. Proper species assignments (different behaviors on different species)
3. Balanced particle counts (total < 5000)
4. Appropriate color palette for the artistic concept
5. Performance-optimized parameters

Also generate the complete Python code following the EXACT Tölvera API patterns with @ti.kernel decorators."""
            )
            
            # Generate the Python code from the configuration
            if hasattr(result.data, 'config'):
                python_code = self._generate_python_code(result.data.config)
                
                # Update the result with generated code
                return SketchGenerationResponse(
                    config=result.data.config,
                    python_code=python_code,
                    explanation=result.data.explanation,
                    suggestions=result.data.suggestions,
                    performance_notes=result.data.performance_notes
                )
            else:
                raise ValueError("Agent did not return valid configuration")
                
        except Exception as e:
            print(f"❌ Error generating sketch: {e}")
            # Return a fallback template
            template = SketchTemplates.organic_flocking_with_trails()
            return SketchGenerationResponse(
                config=template,
                python_code=self._generate_python_code(template),
                explanation=f"Generated fallback organic flocking sketch due to error: {e}",
                suggestions=["Try a simpler description", "Check Ollama connection"],
                performance_notes=["Using safe default parameters"]
            )

    async def modify_sketch(
        self, 
        current_config: MultiBehaviorSketchConfig, 
        modification_request: str
    ) -> SketchModificationResponse:
        """
        Modify an existing sketch based on user feedback.
        
        Args:
            current_config: Current sketch configuration
            modification_request: Description of desired changes
            
        Returns:
            SketchModificationResponse with updated configuration and code
        """
        print(f"🔧 Modifying sketch: {modification_request}")
        
        try:
            # Convert config to dict for the agent to understand what is happening
            config_dict = current_config.model_dump()
            
            result = await self.modification_agent.run(
                f"""Modify this existing Tölvera sketch configuration based on the request: "{modification_request}"

Current configuration: {config_dict}

Generate an updated MultiBehaviorSketchConfig that incorporates the requested changes while maintaining artistic coherence and technical correctness."""
            )
            
            # Generate updated Python code
            if hasattr(result.data, 'updated_config'):
                updated_code = self._generate_python_code(result.data.updated_config)
                
                return SketchModificationResponse(
                    updated_config=result.data.updated_config,
                    updated_code=updated_code,
                    changes_made=result.data.changes_made,
                    explanation=result.data.explanation
                )
            else:
                raise ValueError("Agent did not return valid updated configuration")
                
        except Exception as e:
            print(f"❌ Error modifying sketch: {e}")
            # Return original with error note
            return SketchModificationResponse(
                updated_config=current_config,
                updated_code=self._generate_python_code(current_config),
                changes_made=[f"No changes applied due to error: {e}"],
                explanation="Modification failed, returning original configuration"
            )

    def _generate_python_code(self, config: MultiBehaviorSketchConfig) -> str:
        """
        Generate complete Tölvera Python code from a configuration.
        
        This creates properly structured code with @ti.kernel decorators
        for all state assignments, following Tölvera's exact API requirements.
        """
        
        imports = [
            "from tolvera import Tolvera, run",
            "import taichi as ti  # Required for @ti.kernel decorators"
        ]
        
        config_functions = []
        render_calls = []
        
        for i, behavior in enumerate(config.behaviors):
            if behavior.behavior_type == BehaviorType.FLOCK:
                config_functions.append(self._generate_flock_config(behavior, i))
                render_calls.append("tv.v.flock(tv.p)")
            elif behavior.behavior_type == BehaviorType.SLIME:
                config_functions.append(self._generate_slime_config(behavior, i))
                render_calls.append("tv.v.slime(tv.p, tv.s.species())")
            elif behavior.behavior_type == BehaviorType.PARTICLE_LIFE:
                config_functions.append(self._generate_plife_config(behavior, i))
                render_calls.append("tv.v.plife(tv.p)")
        
        main_function = f'''def main(**kwargs):
    """
    {config.description}
    
    Multi-behavior sketch with: {config.get_behavior_summary()}
    Total particles: {config.get_total_particle_count()}
    """
    tv = Tolvera(**kwargs)
    
{chr(10).join(config_functions)}
    
    # Configure all behaviors
{chr(10).join(f"    configure_behavior_{i}()" for i in range(len(config.behaviors)))}
    
    @tv.render
    def _():
        # Background handling
        {self._generate_background_code(config.render_config)}
        
        # Apply all behaviors
{chr(10).join(f"        {call}" for call in render_calls)}
        
        # Render particles
        tv.px.particles(tv.p, tv.s.species(), "{config.behaviors[0].particle_shape.value}")
        
        return tv.px'''
        
        # Combine everything
        code_parts = [
            f'"""',
            f'{config.description}',
            f'',
            f'Multi-behavior Tölvera sketch generated with PydanticAI.',
            f'Behaviors: {", ".join(b.behavior_type.value for b in config.behaviors)}',
            f'"""',
            '',
            '\n'.join(imports),
            '',
            main_function,
            '',
            "if __name__ == '__main__':",
            "    run(main)"
        ]
        
        return '\n'.join(code_parts)

    def _generate_flock_config(self, behavior: BehaviorInstance, index: int) -> str:
        """Generate @ti.kernel function for flocking configuration."""
        config = behavior.flock_config
        species_configs = []
        
        for species_idx in behavior.species_indices:
            species_configs.append(f'''        # Configure flocking for species {species_idx}
        tv.s.flock_s[{species_idx}, {species_idx}].separate = {config.separation_distance}
        tv.s.flock_s[{species_idx}, {species_idx}].align = {config.alignment_strength}
        tv.s.flock_s[{species_idx}, {species_idx}].cohere = {config.cohesion_force}''')
        
        return f'''    @ti.kernel
    def configure_behavior_{index}():
        """Configure flocking behavior parameters."""
{chr(10).join(species_configs)}'''

    def _generate_slime_config(self, behavior: BehaviorInstance, index: int) -> str:
        """Generate @ti.kernel function for slime configuration."""
        config = behavior.slime_config
        species_configs = []
        
        for species_idx in behavior.species_indices:
            species_configs.append(f'''        # Configure slime for species {species_idx}
        tv.s.slime_s[{species_idx}].sense_angle = {config.sense_angle}
        tv.s.slime_s[{species_idx}].sense_dist = {config.sense_dist}
        tv.s.slime_s[{species_idx}].move_angle = {config.move_angle}
        tv.s.slime_s[{species_idx}].move_step = {config.move_step}''')
        
        return f'''    @ti.kernel
    def configure_behavior_{index}():
        """Configure slime behavior parameters."""
{chr(10).join(species_configs)}'''

    def _generate_plife_config(self, behavior: BehaviorInstance, index: int) -> str:
        """Generate @ti.kernel function for particle life configuration."""
        config = behavior.particle_life_config
        species_configs = []
        
        for species_idx in behavior.species_indices:
            species_configs.append(f'''        # Configure particle life for species {species_idx}
        # Note: Particle life parameters may need custom implementation
        # tv.s.plife_s[{species_idx}].attraction = {config.attraction_strength}''')
        
        return f'''    @ti.kernel
    def configure_behavior_{index}():
        """Configure particle life behavior parameters."""
{chr(10).join(species_configs)}'''

    def _generate_background_code(self, render_config: RenderConfig) -> str:
        """Generate background handling code."""
        if render_config.background_behavior == BackgroundBehavior.DIFFUSE:
            return f"tv.px.diffuse({render_config.diffuse_rate})"
        elif render_config.background_behavior == BackgroundBehavior.CLEAR:
            return "tv.px.clear()"
        else:
            return "# No background processing"

    def get_available_models(self) -> List[str]:
        """Get list of available Ollama models."""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model['name'] for model in models]
            return []
        except Exception:
            return []

    def get_recommended_models(self) -> Dict[str, str]:
        """Get recommended models based on which ones are good for tooling on this."""
        return {
            "creative": "qwen2.5:14b",     
            "balanced": "llama3.1:8b",       
            "fast": "llama3.2:3b",         
            "code": "codestral:latest",    
            "large": "qwen2.5:32b",        
        }


# Convenience functions basically
async def generate_tolvera_sketch(
    description: str, 
    model: str = "qwen2.5:14b",
    ollama_host: str = "http://localhost:11434"
) -> SketchGenerationResponse:
    """
    Generate a Tölvera multi-behavior sketch from natural language.
    
    Args:
        description: Natural language description of the desired artwork
        model: Ollama model to use
        ollama_host: Ollama server URL
        
    Returns:
        SketchGenerationResponse with configuration and executable code
        
    Example:
        >>> result = await generate_tolvera_sketch(
        ...     "Create organic flocking birds that leave slime trails behind them"
        ... )
        >>> print(result.python_code)
    """
    agent = TolveraSketchAgent(model, ollama_host)
    return await agent.generate_sketch(description)


async def modify_tolvera_sketch(
    config: MultiBehaviorSketchConfig,
    modification: str,
    model: str = "qwen2.5:14b", 
    ollama_host: str = "http://localhost:11434"
) -> SketchModificationResponse:
    """
    Modify an existing Tölvera sketch based on feedback.
    
    Args:
        config: Current sketch configuration
        modification: Description of desired changes
        model: Ollama model to use
        ollama_host: Ollama server URL
        
    Returns:
        SketchModificationResponse with updated configuration and code
    """
    agent = TolveraSketchAgent(model, ollama_host)
    return await agent.modify_sketch(config, modification)