# In tolvera/llm/agent.py
import logging
import asyncio
from pydantic_ai import Agent, RunContext
# 1. Import the correct Model and Provider classes
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from .definitions import SketchConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tolvera.ai")


# The router agent is just a plain agent. It will be configured at runtime.
sketch_router_agent = Agent()

@sketch_router_agent.tool
async def generate_flock_sketch(ctx: RunContext) -> SketchConfig:
    """
    Use this tool to generate a complete sketch configuration for a "flock" or
    "flocking" simulation, such as boids, birds, or schools of fish.
    """
    user_prompt = ctx.prompt
    logger.info(f"Using 'generate_flock_sketch' tool for prompt: '{user_prompt}'")

    # The fully configured model object is retrieved from the context
    configured_model = ctx.agent.model

    flock_agent = Agent(
        model=configured_model, # Use the passed-in model object
        system_prompt=(
            "You are an assistant that generates a complete JSON configuration for a Tölvera "
            "'flock' simulation. Adhere strictly to the provided Pydantic model structure. "
            "Set sketch_type to 'flock' and ensure slime_config is null."
        ),
        output_type=SketchConfig,
    )
    
    result = await flock_agent.run(user_prompt)
    return result

@sketch_router_agent.tool
async def generate_slime_sketch(ctx: RunContext) -> SketchConfig:
    """
    Use this tool to generate a complete sketch configuration for a "slime" or
    "slime mold" simulation, often inspired by Physarum polycephalum.
    """
    user_prompt = ctx.prompt
    logger.info(f"Using 'generate_slime_sketch' tool for prompt: '{user_prompt}'")
    
    configured_model = ctx.agent.model

    slime_agent = Agent(
        model=configured_model, # Use the passed-in model object
        system_prompt=(
            "You are an assistant that generates a complete JSON configuration for a Tölvera "
            "'slime' simulation. Adhere strictly to the provided Pydantic model structure. "
            "Set sketch_type to 'slime' and ensure flock_config is null."
        ),
        output_type=SketchConfig,
    )

    result = await slime_agent.run(user_prompt)
    return result

async def generate_sketch_config(
    user_input: str,
    ollama_model_name: str = "llama3.1"
) -> SketchConfig | None:
    """
    Generates a Tölvera sketch configuration from natural language.
    """
    logger.info(f"Received request: '{user_input}'")
    try:
        provider = OpenAIProvider(base_url="http://localhost:11434/v1")


        ollama_model = OpenAIModel(
            model_name=ollama_model_name,
            provider=provider
        )

        final_config = await sketch_router_agent.run(
            user_input,
            model=ollama_model
        )
        
        logger.info("Successfully generated and validated sketch configuration.")
        return final_config
    except Exception as e:
        logger.error(f"Failed to generate sketch config: {e}")
        import traceback
        traceback.print_exc()
        return None