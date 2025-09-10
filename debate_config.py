from dataclasses import dataclass
from typing import List, Dict, Any
from project_types import base_config, test_setup

@dataclass
class AgentConfig:
    """Configuration for a single LLM agent in the debate system."""
    name: str
    provider: str  # "openai", "gemini", "ali", "cst"
    model: str
    temperature: float = 0.7
    max_tokens: int = 2000
    reasoning_effort: str = "medium"  # for OpenAI
    verbosity: str = "medium"  # for OpenAI
    enabled: bool = True

@dataclass
class DebateConfig:
    """Configuration for the multi-agent debate system."""
    agents: List[AgentConfig]
    max_debate_rounds: int = 3
    enable_self_adjustment: bool = True
    enable_majority_vote: bool = True
    supervisor_provider: str = "openai"
    supervisor_model: str = "gpt-4"
    enable_visualization: bool = True
    output_path: str = "./debate_results"
    save_detailed_logs: bool = True

# Default debate configurations
def create_default_debate_config() -> DebateConfig:
    """Create a default debate configuration with 3 agents."""
    agents = [
        AgentConfig(
            name="Agent-1",
            provider="openai",
            model="gpt-5-nano",
            temperature=0.7,
            reasoning_effort="high",
            verbosity="medium"
        ),
        AgentConfig(
            name="Agent-2", 
            provider="gemini",
            model="gemini-2.5-flash-lite",
            temperature=0.8
        ),
        AgentConfig(
            name="Agent-3",
            provider="ali",
            model="qwen-flash",
            temperature=0.6
        )
    ]
    
    return DebateConfig(
        agents=agents,
        max_debate_rounds=3,
        enable_self_adjustment=True,
        enable_majority_vote=True,
        supervisor_provider="openai",
        supervisor_model="gpt-4",
        enable_visualization=True,
        output_path="./debate_results",
        save_detailed_logs=True
    )

def create_custom_debate_config(agent_configs: List[Dict[str, Any]]) -> DebateConfig:
    """Create a custom debate configuration from a list of agent dictionaries.
    
    Args:
        agent_configs: List of dictionaries with agent configuration
        Example: [
            {"name": "GPT-5", "provider": "openai", "model": "gpt-5-nano"},
            {"name": "Gemini", "provider": "gemini", "model": "gemini-2.5-flash"},
            {"name": "Qwen", "provider": "ali", "model": "qwen-flash"}
        ]
    """
    agents = []
    for config in agent_configs:
        agent = AgentConfig(
            name=config.get("name", f"Agent-{len(agents)+1}"),
            provider=config.get("provider", "openai"),
            model=config.get("model", "gpt-5-nano"),
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 2000),
            reasoning_effort=config.get("reasoning_effort", "medium"),
            verbosity=config.get("verbosity", "medium"),
            enabled=config.get("enabled", True)
        )
        agents.append(agent)
    
    return DebateConfig(
        agents=agents,
        max_debate_rounds=config.get("max_debate_rounds", 3),
        enable_self_adjustment=config.get("enable_self_adjustment", True),
        enable_majority_vote=config.get("enable_majority_vote", True),
        supervisor_provider=config.get("supervisor_provider", "openai"),
        supervisor_model=config.get("supervisor_model", "gpt-4"),
        enable_visualization=config.get("enable_visualization", True),
        output_path=config.get("output_path", "./debate_results"),
        save_detailed_logs=config.get("save_detailed_logs", True)
    )
