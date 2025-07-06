
"""
Integration layer between PoE behavior system and Tölvera.

This module provides the glue between the PoE expert system and Tölvera's particle system.
"""

from typing import Dict, Any, List
import logging
import taichi as ti
import linecache
import time

from .poe_core import PoEBehaviorSystem, SimpleProgrammaticExpert
from .poe_experts import ExpertManager
from .poe_ollama import PoEExpertSynthesizer

logger = logging.getLogger(__name__)


class TolveraBehaviorAgent:

    def __init__(self, tolvera_instance):
        self.tv = tolvera_instance
        self.poe_system = PoEBehaviorSystem(tolvera_instance)
        self.expert_manager = ExpertManager()

        logger.info(
            f"Initialized TolveraBehaviorAgent with {tolvera_instance.pn} particles")

    def add_expert_from_code(self, name: str, code: str, weight: float = 1.0):
        expert = SimpleProgrammaticExpert(name, code, weight)
        self.poe_system.add_expert(expert)
        self.expert_manager.add_expert(name, expert)
        logger.info(f"Added expert from code: {name}")
        return expert

    async def add_expert_from_description(
            self,
            description: str,
            synthesizer: PoEExpertSynthesizer,
            weight: float = 1.0):

        # Phase 1: Route the description using LLM
        logger.info(f"Phase 1: Routing description: '{description}'")
        category = await synthesizer.analyze_description(description)

        logger.info(f"Routed to category: {category}")

        # Phase 2: Generate state if needed
        state_name = None
        if category in ["PARTICLE_INTERACTION", "STATE_TRACKING"]:
            logger.info(f"Phase 2: Generating state for: '{description}'")
            state_result = await synthesizer.synthesize_state_from_description(
                description, category
            )

            # Execute state creation code
            logger.info(
                f"Executing state creation code for: {state_result['state_name']}")

            # Use linecache approach similar to poe_core.py
            source_name = f"<poe_generated_state_{time.time_ns()}>"
            state_code = state_result["code"]

            # Put the code into linecache
            linecache.cache[source_name] = (
                len(state_code), None,
                [line + '\n' for line in state_code.splitlines()],
                source_name
            )

            # Create a namespace with necessary imports and objects
            namespace = {
                'tv': self.tv,
                'ti': ti,
            }

            # Compile and execute
            code_obj = compile(state_code, source_name, 'exec')
            exec(code_obj, namespace)

            state_name = state_result["state_name"]

            # Store state definition in PoE system
            self.poe_system.add_state_definition(state_result)
            logger.info(f"Successfully created state: {state_name}")

        # Phase 3: Generate appropriate expert
        logger.info(f"Phase 3: Synthesizing expert for: '{description}'")

        result = await synthesizer.synthesize_expert(
            description,
            category,
            state_info=state_result
        )

        expert = SimpleProgrammaticExpert(
            name=result["name"],
            code=result["code"],
            weight=weight,
            expert_type=result["expert_type"]
        )
        expert.metadata["description"] = description
        expert.metadata["category"] = category
        expert.metadata["associated_state"] = state_name

        logger.info(
            f"Generated {result['expert_type']} expert '{result['name']}' for: '{description}'")

        self.poe_system.add_expert(expert)
        self.expert_manager.add_expert(result["name"], expert)

        # Phase 4: Regenerate integration kernel
        logger.info(
            f"Phase 4: Regenerating integration kernel for {len(self.poe_system.experts)} experts")

        await self.poe_system.regenerate_integration_kernel(synthesizer, self.poe_system.state_definitions)

        logger.info(
            f"Successfully added expert {result['name']} and regenerated kernel")
        return expert

    def set_expert_weight(self, expert_name: str, weight: float):
        self.poe_system.set_expert_weight(expert_name, weight)

    def get_expert_info(self) -> List[Dict[str, Any]]:
        return self.poe_system.get_expert_info()

    def clear_all_experts(self):
        self.poe_system.clear_experts()
        self.expert_manager.clear_all()
        logger.info("Cleared all experts from agent")
