"""
Tests the core functionality and generation for the LLM agent.
"""

import asyncio
import logging
from pathlib import Path
import sys
from pydantic import ValidationError

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# I mess this up all the time when changing up the library function and init functions so this is why this is here
def test_imports():
    try:
        from tolvera.llm import SketchAgent
        from tolvera.llm.models import SketchTemplates, BehaviorType
        from tolvera.llm.generation_prompts import GENERATION_SYSTEM_PROMPT
        from tolvera.llm.code_generation import JinjaCodeGenerator
        from tolvera.llm.utils import get_best_available_model, check_ollama_connection        
        
        logger.info("✅ All imports successful")
        
        return True
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False

def test_code_generation():
    try:
        from tolvera.llm.models import SketchTemplates, BehaviorType, BehaviorInstance, AttractForceConfig
        from tolvera.llm.code_generation import JinjaCodeGenerator
        from tolvera.llm.code_generation.templates import get_template_info
        
        generator = JinjaCodeGenerator()
        template = SketchTemplates.organic_flocking_with_trails()
        code = generator.generate_python_code(template)
        
        # These should never be missing anymore with the template, but when I was trying to just generate the code itself with the agent...nothing was able to compile so that's a problem :)
        required = ["@ti.kernel", "@tv.render"]
        for req in required:
            if req not in code: raise ValueError(f"Missing: {req}")
        
        force_template = SketchTemplates.organic_flocking_with_trails()
        force_template.behaviors.append(
            BehaviorInstance(
                behavior_type=BehaviorType.ATTRACT, particle_count=0,
                species_indices=[0], attract_config=AttractForceConfig()
            )
        )
        force_code = generator.generate_python_code(force_template)
        logger.info("✅ Force function generation working" if "tv.v.attract" in force_code else "⚠️ Force function generation may not be working")

        template_info = get_template_info()
        if not template_info: raise ValueError("No template info available")
        
        logger.info(f"✅ Code generation successful ({len(code.splitlines())} lines)")
        
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        (output_dir / "example_generated.py").write_text(code)
        (output_dir / "example_with_forces.py").write_text(force_code)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Code generation failed: {e}")
        return False

async def test_generation():
    """Test generation with natural descriptions."""
    try:
        from tolvera.llm.utils import check_ollama_connection, get_best_available_model
        from tolvera.llm import SketchAgent
        
        if not check_ollama_connection():
            logger.warning("⚠️ Ollama not running - skipping generation test")
            return True

        model_name = get_best_available_model()
        if not model_name:
            logger.warning("⚠️ No compatible models found - skipping generation test")
            return True
        logger.info(f"✅ Using preferred model: {model_name}")
        
        agent = SketchAgent(model_name=model_name)
        
        natural_descriptions = [
            "A peaceful forest scene with gentle movement",
            "Ocean waves with organic flow patterns", 
            "Busy city intersection with flowing traffic",
            "Solar system with gravitational forces"
        ]
        
        success_count = 0
        for i, description in enumerate(natural_descriptions):
            logger.info(f"Testing natural description {i+1}: '{description}'")
            
            try:
                result = await agent.generate_sketch(description)
                
                if result.config and result.python_code:
                    
                    output_dir = Path("test_output_files")
                    output_dir.mkdir(exist_ok=True)
                    filename = f"natural_gen_{i+1}.py"
                    (output_dir / filename).write_text(result.python_code)
                    logger.info(f"Saved: {filename}")
                    success_count += 1
                else:
                    logger.error(f"Generation for '{description}' failed to produce a valid config.")

            except ValidationError as e:
                logger.error(f"❌ Pydantic Validation Failed for '{description}':")
                logger.error(f"Detailed Errors: {e.errors()}")
                continue
            except Exception as e:
                logger.warning(f"Generation {i+1} failed with other error: {str(e)[:150]}")
                continue
        
        logger.info(f"✅ Generation successful ({success_count}/{len(natural_descriptions)})")
        return True

    except Exception as e:
        logger.error(f"❌ Generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_modification():
    try:
        from tolvera.llm.utils import check_ollama_connection, get_best_available_model
        from tolvera.llm import SketchAgent
        from tolvera.llm.models import SketchTemplates
        
        if not check_ollama_connection():
            logger.warning("⚠️ Ollama not running - skipping modification test")
            return True
        
        # The "best" comes from what works with the tool chain in ollama mainly
        model_name = get_best_available_model()
        if not model_name:
            logger.warning("⚠️ No compatible models found - skipping modification test")
            return True
        logger.info(f"✅ Using preferred model: {model_name}")

        agent = SketchAgent(model_name=model_name)
        original = SketchTemplates.organic_flocking_with_trails()
        modification = "Make it feel more energetic and add some gravitational attraction"
        logger.info(f"Testing modification: '{modification}'")
        
        try:
            result = await agent.modify_sketch(original, modification)
            
            if result.updated_config and result.updated_code:
                behaviors = [b.behavior_type.value for b in result.updated_config.behaviors]
                logger.info("✅ Modification successful")
                logger.info(f"Updated behaviors: {', '.join(behaviors)}")
                logger.info(f"Changes: {', '.join(result.changes_made)}")
                
                output_dir = Path("test_output")
                output_dir.mkdir(exist_ok=True)
                (output_dir / "modified.py").write_text(result.updated_code)
                return True
            else:
                logger.warning("⚠️ Modification returned empty result")
                return True

        except ValidationError as e:
            logger.error("❌ Pydantic Validation Failed during modification:")
            logger.error(f"Detailed Errors: {e.errors()}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Modification test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_simplified_tests():
    
    tests = [
        ("Core Imports", test_imports),
        ("Code Generation", test_code_generation),
        ("Generation", test_generation),
        ("Modification", test_modification)
    ]
    
    passed_count = 0
    total_tests = len(tests)
    
    for name, func in tests:
        logger.info(f"\n {name}")
        logger.info("*" * 40)
        try:
            # Hangouts happen occasionally, so a timeout helps the test keep running if we get hung up on something.
            if asyncio.iscoroutinefunction(func):
                success = await asyncio.wait_for(func(), timeout=120.0)
            else:
                success = func()
            if success:
                passed_count += 1
                logger.info(f"✅ {name} PASSED")
            else:
                logger.error(f"❌ {name} FAILED")
        except asyncio.TimeoutError:
            logger.error(f"❌ {name} FAILED: Test timed out after 120 seconds.")
        except Exception as e:
            logger.error(f"❌ {name} FAILED with unhandled exception: {type(e).__name__} - {e}")
            import traceback
            traceback.print_exc()

    logger.info(f"\n📊 Final Results: {passed_count}/{total_tests} tests passed")
    
    if passed_count == total_tests:
        print("\n✅ Good to go!")
    else:
        print("\n❌ Some tests failed - check the logs above")


if __name__ == "__main__":
    # If you can, make sure ollama is running in the background before executing the test suite.
    
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    asyncio.run(run_simplified_tests())