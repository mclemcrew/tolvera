#!/usr/bin/env python3
import asyncio
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from compositional.orchestrator import CodeGenerationOrchestrator
from compositional.agents import (
    ConductorAgent,
    ParticleCreationAgent,
    ColorAgent,
    MotionAgent,
    PhysicsAgent,
)
# Utils for saving generated scripts
from compositional.utils import save_generated_script

# Setup logging
logging.basicConfig(level=logging.INFO)
# USE DEBUG if you want more info from this
logger = logging.getLogger(__name__)

async def test_individual_agents():
    
    # Test simple requests that should work
    test_cases = [
        ("Conductor", ConductorAgent(), "plan_task", "three blue particles bouncing around"),
        ("Particle", ParticleCreationAgent(), "execute_task", "Create 3 particles positioned randomly"),
        ("Color", ColorAgent(), "execute_task", "Set particle color to blue"),
        ("Motion", MotionAgent(), "execute_task", "Move particles from left to right"),
        ("Physics", PhysicsAgent(), "execute_task", "Apply bouncing behavior to particles"),
    ]
    
    for agent_name, agent, method_name, task in test_cases:
        print(f"\nTesting {agent_name} Agent")
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

async def test_simple_generation():
    print("\n Testing Simple Generation")
    print("=" * 40)
    
    requests = [
        "create three green particles that all move to the right in a straight line and have different velocities",
        "create five orange circles that bounce around the screen very quickly!",
        "create 100 square particles that are crimson in color and bounce around the screen but repel one another when they get close to each other", # This one fails a lot so just heads up on that
    ]
    
    for i, request in enumerate(requests):
        print(f"\nTest {i+1}: {request}")
        
        try:
            orchestrator = CodeGenerationOrchestrator()
            script = await orchestrator.generate_script(request)
            
            print(f"✅ Generated: {script.title}")
            print(f"Code length: {len(script.code)} chars")
            
            # Check for correct syntax
            if "ti.Vector([" in script.code and "ti.Vector(ti.random()" not in script.code:
                print("✅ Contains correct ti.Vector syntax")
            else:
                print("❌ Contains invalid ti.Vector syntax")
            
            # Save for inspection
            filename = f"debug_test_{i+1}.py"
            save_generated_script(script.code, filename)
            print(f"Saved to: {filename}")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True

async def test_agent_routing():
    """Test if agents are being routed correctly."""
    print("\n Testing Agent Routing")
    print("=" * 40)
    
    orchestrator = CodeGenerationOrchestrator()
    
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

async def main():
    print(" Simplified MoE Debug Suite")
    print("=" * 60)
    
    tests = [
        ("Simple Generation", test_simple_generation),
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