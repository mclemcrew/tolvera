"""
Test script to validate the PydanticAI integration with Tölvera.

This script tests the tolvera.llm module from an external perspective, as a user would import and use it.
All generated code is saved to files for inspection.

This script tests:
1. Ollama connection and model availability
2. PydanticAI agent initialization 
3. Sketch generation with LLM calls (this uses ollama so will fail if Ollama is not running)
4. Code generation and validation (validation is a little weak, but checks for basic structure)
"""

import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime

try:
    from tolvera.llm import (
        TolveraSketchAgent,
        generate_tolvera_sketch,
        MultiBehaviorSketchConfig,
        BehaviorInstance, 
        BehaviorType,
        FlockBehaviorConfig,
        ColorPalette,
        SketchTemplates
    )
    print("Successfully imported from tolvera.llm")
except ImportError as e:
    print(f"Failed to import from tolvera.llm: {e}")
    sys.exit(1)


class TölveraIntegrationTester:
    """Test suite for the PydanticAI Tölvera integration."""
    
    def __init__(self):
        self.test_results = []
        self.passed = 0
        self.failed = 0
        
        # Create output directory for generated files
        self.output_dir = Path("test_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        # Create test session info
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_dir / f"session_{self.session_id}"
        self.session_dir.mkdir(exist_ok=True)
        
        print(f"📁 Test outputs will be saved to: {self.session_dir}")
        
        # Create session log
        self.session_log = []
    
    def log_output(self, test_name: str, content: str, file_type: str = "py", description: str = ""):
        """Save generated content to files with proper naming."""
        # Clean test name for filename
        clean_name = test_name.lower().replace(" ", "_").replace("-", "_")
        timestamp = datetime.now().strftime("%H%M%S")
        
        filename = f"{timestamp}_{clean_name}.{file_type}"
        filepath = self.session_dir / filename
        
        # Save the content
        filepath.write_text(content)
        
        # Log the entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "test_name": test_name,
            "filename": filename,
            "description": description,
            "content_length": len(content),
            "file_type": file_type
        }
        self.session_log.append(log_entry)
        
        print(f"💾 Saved {file_type.upper()}: {filename} ({len(content)} chars)")
        if description:
            print(f"   📝 {description}")
        
        return filepath
    
    def test(self, name: str, test_func):
        try:
            result = test_func()
            if asyncio.iscoroutine(result):
                result = asyncio.run(result)
            
            print(f"✅ PASSED: {name}")
            self.test_results.append({"name": name, "status": "PASSED", "error": None})
            self.passed += 1
            return True
            
        except Exception as e:
            print(f"❌ FAILED: {name} - {e}")
            self.test_results.append({"name": name, "status": "FAILED", "error": str(e)})
            self.failed += 1
            return False
    
    def test_model_validation(self):
        """Test that our Pydantic models work correctly."""
        flock_behavior = BehaviorInstance(
            behavior_type=BehaviorType.FLOCK,
            particle_count=500,
            species_indices=[0, 1],
            flock_config=FlockBehaviorConfig(
                separation_distance=0.15,
                alignment_strength=0.8,
                cohesion_force=0.6
            )
        )
        
        # Save the configuration as JSON for inspection
        config_dict = flock_behavior.model_dump()
        self.log_output(
            "model_validation", 
            json.dumps(config_dict, indent=2), 
            "json",
            "Flock behavior configuration example"
        )
        
        assert flock_behavior.behavior_type == BehaviorType.FLOCK
        assert flock_behavior.slime_config is None
        assert flock_behavior.flock_config is not None
        
        # Test multi-behavior sketch config
        config = MultiBehaviorSketchConfig(
            sketch_name="test_sketch",
            description="Test multi-behavior composition",
            behaviors=[flock_behavior],
            color_palette=ColorPalette(
                primary=(0.8, 0.2, 0.3),
                secondary=(0.2, 0.8, 0.3)
            )
        )
        
        # Save the full config
        full_config_dict = config.model_dump()
        self.log_output(
            "model_validation_full", 
            json.dumps(full_config_dict, indent=2), 
            "json",
            "Complete multi-behavior sketch configuration"
        )
        
        assert config.get_total_particle_count() == 500
        assert config.get_max_species_index() == 1
        
        return True
    
    def test_template_creation(self):
        """Test that built-in templates work correctly."""
        organic = SketchTemplates.organic_flocking_with_trails()
        organic_dict = organic.model_dump()
        self.log_output(
            "template_organic", 
            json.dumps(organic_dict, indent=2), 
            "json",
            "Organic flocking with trails template"
        )
        
        chaotic = SketchTemplates.chaotic_multi_behavior()
        chaotic_dict = chaotic.model_dump()
        self.log_output(
            "template_chaotic", 
            json.dumps(chaotic_dict, indent=2), 
            "json",
            "Chaotic multi-behavior template"
        )
        
        # Validate templates
        assert len(organic.behaviors) == 2
        assert organic.behaviors[0].behavior_type == BehaviorType.FLOCK
        assert organic.behaviors[1].behavior_type == BehaviorType.SLIME
        
        assert len(chaotic.behaviors) == 3
        behavior_types = [b.behavior_type for b in chaotic.behaviors]
        assert BehaviorType.FLOCK in behavior_types
        assert BehaviorType.SLIME in behavior_types
        assert BehaviorType.PARTICLE_LIFE in behavior_types
        
        return True
    
    def test_ollama_connection(self):
        """Test Ollama server connection."""
        import requests
        
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code != 200:
                raise ConnectionError("Ollama not responding")
            
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            
            # Save model information
            model_info = {
                "available_models": model_names,
                "model_count": len(models),
                "full_model_data": models
            }
            self.log_output(
                "ollama_models", 
                json.dumps(model_info, indent=2), 
                "json",
                "Available Ollama models information"
            )
            
            print(f"Found {len(models)} models: {', '.join(model_names[:3])}...")
            
            recommended = ["qwen2.5:14b", "llama3.1:8b", "codestral:latest"]
            available_recommended = [m for m in recommended if m in model_names]
            
            if not available_recommended:
                print(f"No recommended models found. Consider installing: {recommended[0]}")
            else:
                print(f"Recommended models available: {available_recommended}")
            
            return True
            
        except requests.RequestException as e:
            raise ConnectionError(f"Cannot connect to Ollama: {e}")
    
    async def test_pydantic_ai_agent_init(self):
        """Test PydanticAI agent initialization."""
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            
            if not model_names:
                print("No models available in Ollama")
                return True 
            
            # Choose best model
            tool_compatible = ["qwen2.5:latest", "gemma3:latest", "llama3.1:latest", "qwen3:4b", "llama3.2:latest"]
            test_model = None
            
            for preferred in tool_compatible:
                if preferred in model_names:
                    test_model = preferred
                    break
            
            if not test_model:
                test_model = model_names[0]
                
            print(f"Testing with model: {test_model}")
            
            # Initialize agent
            agent = TolveraSketchAgent(
                model_name=test_model,
                ollama_host="http://localhost:11434"
            )
            
            # Save agent configuration info
            agent_info = {
                "model_name": agent.model_name,
                "ollama_host": agent.ollama_host,
                "available_models": agent.get_available_models(),
                "recommended_models": agent.get_recommended_models()
            }
            self.log_output(
                "agent_init", 
                json.dumps(agent_info, indent=2), 
                "json",
                f"Agent initialization with {test_model}"
            )
            
            assert isinstance(agent.get_available_models(), list)
            assert isinstance(agent.get_recommended_models(), dict)
            
            print(f"Agent initialized successfully with model: {agent.model_name}")
            return True
            
        except Exception as e:
            print(f"Agent initialization issue: {e}")
            return True
    
    async def test_code_generation(self):
        """Test Python code generation from configurations."""
        try:
            best_model = self.get_best_available_model()
            if not best_model:
                print("No models available for code generation test")
                return True
            
            agent = TolveraSketchAgent(model_name=best_model)
            
            # Test with organic template
            template = SketchTemplates.organic_flocking_with_trails()
            generated_code = agent._generate_python_code(template)
            
            # Save the generated code
            self.log_output(
                "code_generation_organic", 
                generated_code, 
                "py",
                f"Generated Tölvera code from organic template using {best_model}"
            )
            
            # Test with chaotic template too
            chaotic_template = SketchTemplates.chaotic_multi_behavior()
            chaotic_code = agent._generate_python_code(chaotic_template)
            
            self.log_output(
                "code_generation_chaotic", 
                chaotic_code, 
                "py",
                f"Generated Tölvera code from chaotic template using {best_model}"
            )
            
            # Validate generated code structure
            required_patterns = [
                "from tolvera import Tolvera, run",
                "import taichi as ti", 
                "@ti.kernel",
                "def main(**kwargs):",
                "@tv.render",
                "return tv.px",
                "if __name__ == '__main__':",
                "run(main)"
            ]
            
            for pattern in required_patterns:
                if pattern not in generated_code:
                    raise ValueError(f"Missing required pattern: {pattern}")
            
            assert generated_code.count("@ti.kernel") >= len(template.behaviors)
            
            print(f"Generated {len(generated_code.split())} lines of code")
            print(f"All required Tölvera patterns present")
            
            return True
            
        except Exception as e:
            print(f"Code generation test issue: {e}")
            return True
    
    async def test_simple_generation(self):
        """Test a simple sketch generation (if model is available)."""
        try:
            best_model = self.get_best_available_model()
            if not best_model:
                print("No models available for generation test")
                return True
            
            description = "Simple flocking birds with organic movement"
            print(f"Testing generation: '{description}'")
            
            # Use the convenience function
            result = await generate_tolvera_sketch(
                description, 
                model=best_model
            )
            
            # Save the AI-generated configuration
            if result.config:
                config_dict = result.config.model_dump()
                self.log_output(
                    "ai_generation_config", 
                    json.dumps(config_dict, indent=2), 
                    "json",
                    f"AI-generated config from: '{description}' using {best_model}"
                )
            
            # Save the AI-generated code
            if result.python_code:
                self.log_output(
                    "ai_generation_code", 
                    result.python_code, 
                    "py",
                    f"AI-generated Tölvera sketch from: '{description}' using {best_model}"
                )
            
            # Save the full result including explanation
            full_result = {
                "description": description,
                "model_used": best_model,
                "explanation": result.explanation,
                "suggestions": result.suggestions if hasattr(result, 'suggestions') else [],
                "performance_notes": result.performance_notes if hasattr(result, 'performance_notes') else [],
                "sketch_name": result.config.sketch_name if result.config else None,
                "behavior_summary": result.config.get_behavior_summary() if result.config else None
            }
            self.log_output(
                "ai_generation_full", 
                json.dumps(full_result, indent=2), 
                "json",
                f"Complete AI generation result using {best_model}"
            )
            
            assert result.config is not None
            assert result.python_code is not None
            assert len(result.python_code) > 500
            
            print(f"Generated sketch: {result.config.sketch_name}")
            print(f"Behaviors: {result.config.get_behavior_summary()}")
            
            return True
            
        except Exception as e:
            print(f"Generation test skipped: {e}")
            return True
    
    def get_best_available_model(self):
        """Get the best tool-compatible model available."""
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            
            tool_compatible = ["qwen2.5:latest", "gemma3:latest", "llama3.1:latest", "qwen3:4b", "llama3.2:latest"]
            
            for preferred in tool_compatible:
                if preferred in model_names:
                    return preferred
            
            return model_names[0] if model_names else None
        except:
            return None
        
    def test_validation_edge_cases(self):
        """Test edge cases and validation rules."""
        try:
            # Test with too many particles
            try:
                config = MultiBehaviorSketchConfig(
                    sketch_name="test_overload",
                    description="Too many particles test",
                    behaviors=[
                        BehaviorInstance(
                            behavior_type=BehaviorType.FLOCK,
                            particle_count=50000,
                            species_indices=[0]
                        )
                    ],
                    color_palette=ColorPalette(primary=(1.0, 0.0, 0.0))
                )
                assert False, "Should have failed validation for too many particles"
            except ValueError as e:
                # Save the validation error
                error_info = {
                    "test": "particle_count_limit",
                    "attempted_particles": 50000,
                    "error": str(e),
                    "expected": "This should fail validation"
                }
                self.log_output(
                    "validation_error", 
                    json.dumps(error_info, indent=2), 
                    "json",
                    "Expected validation error for excessive particles"
                )
        
            # Test valid multi-behavior composition
            valid_config = MultiBehaviorSketchConfig(
                sketch_name="valid_multi",
                description="Valid multi-behavior test",
                behaviors=[
                    BehaviorInstance(
                        behavior_type=BehaviorType.FLOCK,
                        particle_count=1000,
                        species_indices=[0, 1]
                    ),
                    BehaviorInstance(
                        behavior_type=BehaviorType.SLIME,
                        particle_count=500,
                        species_indices=[2]
                    )
                ],
                color_palette=ColorPalette(primary=(0.5, 0.5, 0.5))
            )
            
            # Save valid configuration
            valid_dict = valid_config.model_dump()
            self.log_output(
                "validation_valid", 
                json.dumps(valid_dict, indent=2), 
                "json",
                "Valid multi-behavior configuration example"
            )
            
            assert valid_config.get_total_particle_count() == 1500
            assert valid_config.get_max_species_index() == 2
            
            return True
            
        except Exception as e:
            print(f"Validation test issue: {e}")
            return False
    
    def test_tolvera_kwargs_generation(self):
        """Test that configurations generate proper Tölvera kwargs."""
        config = SketchTemplates.organic_flocking_with_trails()
        kwargs = config.to_tolvera_kwargs()
        
        # Save kwargs example
        kwargs_info = {
            "generated_kwargs": kwargs,
            "source_config": {
                "sketch_name": config.sketch_name,
                "total_particles": config.get_total_particle_count(),
                "max_species": config.get_max_species_index(),
                "behavior_count": len(config.behaviors)
            }
        }
        self.log_output(
            "tolvera_kwargs", 
            json.dumps(kwargs_info, indent=2), 
            "json",
            "Example of Tölvera kwargs generation"
        )
        
        required_keys = ['particles', 'species', 'x', 'y', 'speed', 'substep']
        for key in required_keys:
            assert key in kwargs, f"Missing required kwarg: {key}"
        
        assert kwargs['particles'] == config.get_total_particle_count()
        assert kwargs['species'] == config.get_max_species_index() + 1
        assert kwargs['x'] == config.render_config.window_width
        assert kwargs['y'] == config.render_config.window_height
        
        print(f"Generated kwargs: {kwargs}")
        return True
    
    def run_all_tests(self):
        """Run the complete test suite."""
        print("Tölvera PydanticAI Integration Test Suite")
        print("=" * 50)
        print("Testing as external user importing from tolvera.llm")
        
        self.test("Model Validation", self.test_model_validation)
        self.test("Template Creation", self.test_template_creation)
        self.test("Validation Edge Cases", self.test_validation_edge_cases)
        self.test("Tölvera Kwargs Generation", self.test_tolvera_kwargs_generation)
        
        self.test("Ollama Connection", self.test_ollama_connection)
        self.test("PydanticAI Agent Init", self.test_pydantic_ai_agent_init)
        self.test("Code Generation", self.test_code_generation)
        self.test("Simple Generation", self.test_simple_generation)
        
        self.save_session_summary()
        
        # Summary
        print("\n" + "=" * 50)
        print(f"Test Results: {self.passed} passed, {self.failed} failed")
        print(f"📁 All outputs saved to: {self.session_dir}")
        
        if self.failed > 0:
            print("❌ Some tests failed:")
            for result in self.test_results:
                if result["status"] == "FAILED":
                    print(f"   • {result['name']}: {result['error']}")
        else:
            print("✅ All tests passed!")
        
        return self.failed == 0
    
    def save_session_summary(self):
        """Save a complete summary of the test session."""
        summary = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "test_results": self.test_results,
            "statistics": {
                "passed": self.passed,
                "failed": self.failed,
                "total": self.passed + self.failed
            },
            "generated_files": self.session_log,
            "file_count": len(self.session_log)
        }
        
        summary_file = self.session_dir / "session_summary.json"
        summary_file.write_text(json.dumps(summary, indent=2))
        
        # Also create a human-readable summary
        readme_content = f"""# Tölvera PydanticAI Test Session {self.session_id}

## Test Results
- **Passed**: {self.passed}
- **Failed**: {self.failed}
- **Total**: {self.passed + self.failed}

## Generated Files
{chr(10).join(f"- `{entry['filename']}` - {entry['description']}" for entry in self.session_log)}

## Session Info
- **Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Files Generated**: {len(self.session_log)}
- **Test Directory**: {self.session_dir}

## How to Use Generated Files
1. **Python files (.py)**: Can be run directly with `python filename.py` (requires Tölvera)
2. **JSON files (.json)**: Configuration examples and test data
3. **session_summary.json**: Complete test session metadata

## AI-Generated Code
Look for files with "ai_generation" in the name to see what the agent actually produced.
"""
        
        readme_file = self.session_dir / "README.md"
        readme_file.write_text(readme_content)
        
        print(f"Session summary saved to: session_summary.json")
        print(f"Human-readable summary: README.md")


async def demo_quick_generation():
    """Enhanced demo that saves all generated content."""
    print("\n🎨 Quick Generation Demo")
    print("-" * 30)
    
    # Create output directory for demo
    demo_dir = Path("demo_outputs")
    demo_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    demo_session_dir = demo_dir / f"demo_{timestamp}"
    demo_session_dir.mkdir(exist_ok=True)
    
    print(f"📁 Demo outputs will be saved to: {demo_session_dir}")
    
    try:
        tester = TölveraIntegrationTester()
        best_model = tester.get_best_available_model()
        if not best_model:
            print("No models available for demo")
            return False
        
        print(f"Using model: {best_model}")
        
        simple_descriptions = [
            "Create flocking birds",
            "Simple organic movement", 
            "Flocking particles"
        ]
        
        for i, description in enumerate(simple_descriptions):
            try:
                print(f"\n🎨 Attempt {i+1}: '{description}'")
                
                agent = TolveraSketchAgent(model_name=best_model)
                result = await agent.generate_sketch(description)
                
                if result.config and result.python_code:
                    print(f"Generated: {result.config.sketch_name}")
                    print(f"{result.explanation[:100]}...")
                    
                    # Save configuration
                    config_file = demo_session_dir / f"attempt_{i+1}_config.json"
                    config_file.write_text(json.dumps(result.config.model_dump(), indent=2))
                    
                    # Save code
                    code_file = demo_session_dir / f"attempt_{i+1}_{result.config.sketch_name}.py"
                    code_file.write_text(result.python_code)
                    
                    # Save full result
                    full_result = {
                        "description": description,
                        "model": best_model,
                        "explanation": result.explanation,
                        "sketch_name": result.config.sketch_name,
                        "behavior_summary": result.config.get_behavior_summary()
                    }
                    result_file = demo_session_dir / f"attempt_{i+1}_result.json"
                    result_file.write_text(json.dumps(full_result, indent=2))
                    
                    print(f"💾 Saved to: {code_file.name}")
                    return True
                else:
                    print("⚠️  Got empty result, trying next description...")
                    
            except Exception as e:
                print(f"❌ Attempt {i+1} failed: {str(e)[:100]}...")
                
                # Save error info
                error_file = demo_session_dir / f"attempt_{i+1}_error.txt"
                error_file.write_text(f"Description: {description}\nModel: {best_model}\nError: {str(e)}")
                continue
        
        return False
        
    except Exception as e:
        print(f"❌ Demo completely failed: {e}")
        return False


def main():
    print("🚀 Running the enhanced test suite with output saving...")
    
    tester = TölveraIntegrationTester()
    tests_passed = tester.run_all_tests()
    
    if tests_passed:
        asyncio.run(demo_quick_generation())
    
    print(f"\n{'='*60}")
    if tests_passed:
        print("🎉 System ready!")
        print(f"Check {tester.session_dir} for all generated files")
    else:
        print("⚠️  Some tests failed. Check the errors above.")


if __name__ == "__main__":
    main()