from dataclasses import dataclass
from typing import List, Dict, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_types import base_config, test_setup

@dataclass
class AgentConfig:
    """Configuration for a single LLM agent in the debate system."""
    name: str
    provider: str  # "openai", "gemini", "ali", "cst"
    model: str
    temperature: float = 0.1
    max_tokens: int = 500
    reasoning_effort: str = "medium"  # for OpenAI
    verbosity: str = "low"  # for OpenAI
    enabled: bool = True

@dataclass
class DebateConfig:
    """Configuration for the multi-agent debate system."""
    agents: List[AgentConfig]
    max_debate_rounds: int = 1
    enable_self_adjustment: bool = True
    enable_majority_vote: bool = True
    supervisor_provider: str = "openai"
    supervisor_model: str = "gpt-5-mini"
    enable_visualization: bool = True
    output_path: str = "./debate_results"
    save_detailed_logs: bool = True
    game_size: int = 4
    game_num: int = 1
    
    def get_organized_output_path(self, game_id: int = None) -> str:
        """Generate organized folder structure based on agent configuration and game parameters."""
        # Create agent folder name: agent3_{model1}_{model2}_{model3}
        agent_count = len(self.agents)
        model_names = [agent.model for agent in self.agents]
        agent_folder = f"agent{agent_count}_{'_'.join(model_names)}"
        
        # Create game folder name: game_size{size}_id{num}
        if game_id is None:
            game_id = self.game_num
        game_folder = f"game_size{self.game_size}_id{game_id}"
        
        # Combine paths
        organized_path = os.path.join(self.output_path, agent_folder, game_folder)
        return organized_path

# Default debate configurations
def create_default_debate_config(game_size: int = 5, game_num: int = 1) -> DebateConfig:
    """Create a default debate configuration with 3 agents."""
    agents = [
        AgentConfig(
            name="Agent-1",
            provider="openai",
            model="gpt-5-nano",
            # temperature=0.7,
            reasoning_effort="low",
            # verbosity="medium"
        ),
        AgentConfig(
            name="Agent-2", 
            provider="openai",
            model="gpt-4.1-mini",
            temperature=0.1
        ),
        AgentConfig(
            name="Agent-3",
            provider="openai",
            model="gpt-4o-mini",
            temperature=0.1
        )
    ]
    
    return DebateConfig(
        agents=agents,
        max_debate_rounds=3,
        enable_self_adjustment=True,
        enable_majority_vote=True,
        supervisor_provider="openai",
        supervisor_model="gpt-5-mini",
        enable_visualization=True,
        output_path="./debate_results",
        save_detailed_logs=True,
        game_size=game_size,
        game_num=game_num
    )

def create_custom_debate_config(agent_configs: List[Dict[str, Any]], game_size: int = 5, game_num: int = 1) -> DebateConfig:
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
        max_debate_rounds=config.get("max_debate_rounds", 1),
        enable_self_adjustment=config.get("enable_self_adjustment", True),
        enable_majority_vote=config.get("enable_majority_vote", True),
        supervisor_provider=config.get("supervisor_provider", "openai"),
        supervisor_model=config.get("supervisor_model", "gpt-5-mini"),
        enable_visualization=config.get("enable_visualization", True),
        output_path=config.get("output_path", "./debate_results"),
        save_detailed_logs=config.get("save_detailed_logs", True),
        game_size=game_size,
        game_num=game_num
    )

