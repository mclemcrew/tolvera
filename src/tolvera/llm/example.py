#!/usr/bin/env python3
"""
Simplified test to debug specific MoE issues.
"""

import asyncio
import logging

from .compositional.orchestrator import RobustCodeGenerationOrchestrator
from .compositional.agents import (
    RobustConductorAgent,
    RobustParticleCreationAgent,
    RobustColorPaletteAgent,
    RobustMotionDynamicsAgent,
    RobustPhysicsAgent,
)
from .compositional.json_utils import (
    clean_json_response,
    safe_json_parse,
)
from .compositional.utils import (
    validate_generated_script,
    save_generated_script,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_individual_agents():
    """Test each agent individually to isolate issues."""
    print("🧪 Testing Individual Agents (Debug Mode)")
    print("=" * 60)
    
    # Test simple requests that should work
    test_cases = [
        ("Conductor", RobustConductorAgent(), "plan_task", "three blue particles bouncing around"),
        ("Particle", RobustParticleCreationAgent(), "execute_task", "Create 3 particles positioned randomly"),
        ("Color", RobustColorPaletteAgent(), "execute_task", "Set particle color to blue"),
        ("Motion", RobustMotionDynamicsAgent(), "execute_task", "Move particles from left to right"),
        ("Physics", RobustPhysicsAgent(), "execute_task", "Apply bouncing behavior to particles"),
    ]
    
    for agent_name, agent, method_name, task in test_cases:
        print(f"\n🔍 Testing {agent_name} Agent")
        print(f"Task: '{task}'")
        
        try:
            method = getattr(agent, method_name)
            result = await method(task)
            
            print(f"✅ {agent_name}: Success")
            print(f"   Type: {type(result)}")
            
            if hasattr(result, 'explanation'):
                print(f"   Explanation: {result.explanation}")
            if hasattr(result, 'tool_calls'):
                print(f"   Tool calls: {len(result.tool_calls)}")
                for i, call in enumerate(result.tool_calls):
                    print(f"     {i+1}. {call.tool_name}: {call.parameters}")
            if hasattr(result, 'steps'):
                print(f"   Steps: {len(result.steps)}")
                for i, step in enumerate(result.steps):
                    print(f"     {i+1}. {step}")
            
        except Exception as e:
            print(f"❌ {agent_name}: Failed - {e}")
            import traceback
            traceback.print_exc()

async def test_json_issues():
    """Test specific JSON parsing issues."""
    print("\n🧪 Testing JSON Issues")
    print("=" * 40)
    
    # Test the problematic JSON from the logs
    problematic_json = '''{
    "tool_calls": [
        {
            "tool_name": "create_particle",
            "parameters": {"position": [Math.random() * 800, Math.random() * 600]}
        }
    ],
    "explanation": "Created particles"
}'''
    
    print("Testing problematic JSON:")
    print(problematic_json[:100] + "...")
    
    cleaned = clean_json_response(problematic_json)
    parsed = safe_json_parse(problematic_json)
    
    print(f"Cleaned: {cleaned[:100]}...")
    print(f"Parsed: {parsed}")

async def test_simple_generation():
    """Test a single simple generation with detailed logging."""
    print("\n🧪 Testing Simple Generation")
    print("=" * 40)
    
    request = "create three green particles that all move to the right and have different velocities"
    print(f"Request: '{request}'")
    
    try:
        orchestrator = RobustCodeGenerationOrchestrator()
        script = await orchestrator.generate_script(request)
        
        print(f"✅ Generated: {script.title}")
        print(f"Code length: {len(script.code)} chars")
        
        # Validate
        validation = validate_generated_script(script.code)
        print(f"Validation score: {validation['score']}/100")
        
        # Save for inspection
        save_generated_script(script.code, "debug_simple.py")
        print("💾 Saved to: debug_simple.py")
        
        # Show a preview
        print("\n📄 Code Preview (first 20 lines):")
        lines = script.code.split('\n')
        for i, line in enumerate(lines[:20]):
            print(f"{i+1:2}: {line}")
        
        return validation['score'] >= 80
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_agent_routing():
    """Test if agents are being routed correctly."""
    print("\n🧪 Testing Agent Routing")
    print("=" * 40)
    
    orchestrator = RobustCodeGenerationOrchestrator()
    
    test_steps = [
        "Create 3 particles positioned randomly on screen",
        "Set particle color to green", 
        "Apply bouncing physics with random velocities",
        "Assemble final script"
    ]
    
    for i, step in enumerate(test_steps):
        print(f"\nStep {i+1}: '{step}'")
        
        # Test the routing logic
        agent_type = orchestrator._determine_agent_type(step)
        print(f"Routed to: {agent_type}")
        
        # This should show us if routing is working correctly

async def main():
    """Run simplified debugging tests."""
    print("🚀 Simplified MoE Debug Suite")
    print("=" * 60)
    
    tests = [
        ("Simple Generation", test_simple_generation),
        ("JSON Issues", test_json_issues),
        ("Individual Agents", test_individual_agents), 
        ("Agent Routing", test_agent_routing),
    ]
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            await test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())