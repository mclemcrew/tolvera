"""
Pydantic AI version of LLM Agent
"""

import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, ValidationError
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from .models import (
    MultiBehaviorSketchConfig,
    SketchGenerationResponse,
    SketchModificationResponse,
    SketchTemplates
)
from .generation_prompts import GENERATION_SYSTEM_PROMPT, MODIFICATION_SYSTEM_PROMPT, build_generation_prompt, build_modification_prompt
from .code_generation import create_code_generator
from .utils import OllamaModelManager

logger = logging.getLogger(__name__)

class LLMGenerationOutput(BaseModel):
    """
    Defines the expected output from the LLM for sketch generation.
    
    This is used to validate the LLM output against the Pydantic models we have in models.py
    
    For a deeper dive, check out that file.
    """
    config: MultiBehaviorSketchConfig
    explanation: str
    suggestions: Optional[List[str]] = None

class LLMModificationOutput(BaseModel):
    """
    Same vibes as above, but we want the LLM to also explain what it changed and why in response to the user's query.
    """
    updated_config: MultiBehaviorSketchConfig
    changes_made: List[str]
    explanation: str


def _extract_and_clean_json(raw_text: str) -> str:
    """
    Finds a JSON blob within a larger string and extracts it, and cleans it.  This is helpful when these smaller models tend to hallucinate a bit
    """
    # Regex to find a JSON object enclosed in curly braces
    json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    
    if not json_match:
        logger.warning("No JSON object found in the LLM output.")
        return ""
        
    json_str = json_match.group(0)
    
    # Remove trailing commas from objects and lists
    cleaned_str = re.sub(r',\s*([}\]])', r'\1', json_str)
    
    return cleaned_str


class SketchAgent:
    """
    The main agent class.  Could also call this something like "LLMAgent" if we wanted??
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_manager = OllamaModelManager()
        self.model_name = self._select_model(model_name)
        self.ollama_host = "http://localhost:11434" # This is an ollama thing 
        self.code_generator = create_code_generator()
        self._setup_model()
        logger.info(f"SketchAgent initialized with model: {self.model_name}")

    def _select_model(self, requested_model: Optional[str]) -> str:
        try:
            # Based on ollama docs - models that are good with tool calling
            return self.model_manager.ensure_compatible_model(requested_model)
        except Exception as e:
            logger.error(f"❌ Model selection failed: {e}")
            # default based on my setup
            return "qwen3:4b"

    def _setup_model(self):
        # This is pydantic ai's way of pinging the ollama models.  Odd, but just following their structure.
        # The reference link you want: https://ai.pydantic.dev/models/openai/?query=v1#openai-responses-api
        self.model = OpenAIModel(
            model_name=self.model_name,
            provider=OpenAIProvider(
                base_url=f"{self.ollama_host}/v1",
                api_key="ollama"
            )
        )

    async def _run_with_validation(self, prompt: str, result_type: BaseModel, system_prompt: str) -> BaseModel:
        agent = Agent(model=self.model, system_prompt=system_prompt)
        
        for attempt in range(2): 
            raw_response_obj = await agent.run(prompt)
            raw_response_str = raw_response_obj.output
            
            cleaned_json = _extract_and_clean_json(raw_response_str)
            
            if not cleaned_json:
                logger.warning(f"⚠️ Could not extract JSON on attempt {attempt + 1}.")
                if attempt == 0:
                    # Basically blame the model and then try and force it again to do this.
                    prompt = f"Your previous response did not contain a valid JSON object. Please try again. Output only the JSON. Original prompt was: {prompt}"
                    continue
                else:
                    raise ValueError("LLM failed to produce a JSON object after retry.")

            try:
                # Pydantic stuff.  Look to here: https://docs.pydantic.dev/latest/concepts/models/#basic-model-usage
                validated_output = result_type.model_validate_json(cleaned_json)
                return validated_output
            except ValidationError as e:
                logger.warning(f"⚠️ Validation failed on attempt {attempt + 1}.")
                if attempt == 0:
                    # Trying to retry automatically with the new information provided 
                    prompt = f"The previous JSON was invalid. Please fix it based on the following error: {e.errors()}. Original prompt was: {prompt}. Output only the corrected, valid JSON."
                else:
                    raise e
        raise ValueError("Exceeded maximum retries for result validation")


    async def generate_sketch(self, description: str,
                            style: str = "organic",
                            performance_target: str = "balanced") -> SketchGenerationResponse:
        try:
            logger.info(f"Generating sketch: {description}")
            prompt = build_generation_prompt(description, style, performance_target)
            
            llm_output = await self._run_with_validation(prompt, LLMGenerationOutput, GENERATION_SYSTEM_PROMPT)

            if llm_output and llm_output.config:
                python_code = self.code_generator.generate_python_code(llm_output.config)
                
                return SketchGenerationResponse(
                    config=llm_output.config,
                    python_code=python_code,
                    explanation=llm_output.explanation,
                    suggestions=llm_output.suggestions or []
                )
            else:
                raise ValueError("No valid configuration returned from model")

        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            return self._create_fallback_response(description, str(e))

    async def modify_sketch(self, config: MultiBehaviorSketchConfig,
                          modification: str) -> SketchModificationResponse:
        try:
            logger.info(f"Modifying sketch: {modification}")
            prompt = build_modification_prompt(config, modification)
            
            llm_output = await self._run_with_validation(prompt, LLMModificationOutput, MODIFICATION_SYSTEM_PROMPT)

            if llm_output and llm_output.updated_config:
                updated_code = self.code_generator.generate_python_code(llm_output.updated_config)
                
                return SketchModificationResponse(
                    updated_config=llm_output.updated_config,
                    updated_code=updated_code,
                    changes_made=llm_output.changes_made,
                    explanation=llm_output.explanation
                )
            else:
                raise ValueError("No valid updated configuration returned from model")

        except Exception as e:
            logger.error(f"❌ Modification failed: {e}")
            return self._create_fallback_modification(config, str(e))

    def _create_fallback_response(self, description: str, error: str) -> SketchGenerationResponse:
        template = SketchTemplates.organic_flocking_with_trails()
        return SketchGenerationResponse(
            config=template,
            python_code=self.code_generator.generate_python_code(template),
            explanation=f"Fallback sketch. Original error: {error}",
            suggestions=[]
        )

    def _create_fallback_modification(self, config: MultiBehaviorSketchConfig,
                                   error: str) -> SketchModificationResponse:
        return SketchModificationResponse(
            updated_config=config,
            updated_code=self.code_generator.generate_python_code(config),
            changes_made=[f"No changes applied due to error: {error}"],
            explanation="Modification failed, returning original"
        )

    def create_custom_template(self,
                             name: str,
                             description: str,
                             flock_particles: int = 1200,
                             slime_particles: int = 800) -> MultiBehaviorSketchConfig:
        from .models import (BehaviorInstance, FlockBehaviorConfig, SlimeBehaviorConfig,
                           ColorPalette, RenderConfig, ParticleShape, BackgroundBehavior, BehaviorType)
        return MultiBehaviorSketchConfig(
            sketch_name=name,
            description=description,
            behaviors=[
                BehaviorInstance(
                    behavior_type=BehaviorType.FLOCK,
                    particle_count=flock_particles,
                    species_indices=[0, 1],
                    flock_config=FlockBehaviorConfig(separate=0.12, align=0.75, cohere=0.65, radius=0.2),
                    particle_shape=ParticleShape.CIRCLE
                ),
                BehaviorInstance(
                    behavior_type=BehaviorType.SLIME,
                    particle_count=slime_particles,
                    species_indices=[2],
                    slime_config=SlimeBehaviorConfig(sense_angle=0.6, sense_dist=0.5, move_dist=0.2, evaporate=0.95),
                    particle_shape=ParticleShape.POINT
                )
            ],
            color_palette=ColorPalette(primary=(0.2, 0.8, 0.3), secondary=(0.6, 0.9, 0.4), accent=(0.8, 0.4, 0.1)),
            render_config=RenderConfig(background_behavior=BackgroundBehavior.DIFFUSE, diffuse_rate=0.98)
        )