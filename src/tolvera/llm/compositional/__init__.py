from .orchestrator import CodeGenerationOrchestrator, create_orchestrator

from .agents import (
    ConductorAgent,
    ParticleCreationAgent, 
    ColorAgent,
    MotionAgent,
    PhysicsAgent,
    CompositionAgent,
    BaseAgent,
    create_agents,
)

from .tools import (    
    ToolCall,
    TaskResult,
    TaskPlan,
    GeneratedScript,
    STANDARD_COLORS,
)

AGENT_SPECIALIZATIONS = {
    "conductor": "Master planner using structured TaskPlan outputs for task decomposition",
    "particle": "Particle creation expert using structured TaskResult outputs for accurate particle generation",
    "color": "Color management expert using structured TaskResult outputs for comprehensive color handling", 
    "motion": "Movement expert using structured TaskResult outputs for direction and velocity control",
    "physics": "Physics expert using structured TaskResult outputs for organic behaviors like flocking and bouncing",
    "composition": "Script composition expert using LLM-generated GeneratedScript outputs for complete Tölvera code"
}

__all__ = [
    "CodeGenerationOrchestrator",
    "create_orchestrator",
    "GeneratedScript",
    "ToolCall",
    "TaskResult",
    "TaskPlan",
    "ConductorAgent",
    "ParticleCreationAgent",
    "ColorAgent",
    "MotionAgent",
    "PhysicsAgent",
    "CompositionAgent",
    "BaseAgent",
    "create_agents",
    "STANDARD_COLORS",
]