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
    max_tokens: int = 2000
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
    supervisor_model: str = "gpt-5-nano"
    enable_visualization: bool = True
    output_path: str = "./debate_results"
    save_detailed_logs: bool = True
    game_size: int = 4
    game_id_range: List[int] = None  # [start_id, end_id] inclusive
    game_parallel_workers: int = 20  # Number of workers for game-level parallel processing
    
    def __post_init__(self):
        """Set default game_id_range if not provided."""
        if self.game_id_range is None:
            self.game_id_range = [1, 1]  # Default to single game
    
    def get_organized_output_path(self, game_id: int = None) -> str:
        """Generate organized folder structure based on agent configuration and game parameters."""
        # Create agent folder name: agent3_{model1}_{model2}_{model3}
        agent_count = len(self.agents)
        model_names = [agent.model for agent in self.agents]
        agent_folder = f"agent{agent_count}_{'_'.join(model_names)}"
        
        # Create game folder name: game_size{size}_id{num}
        if game_id is None:
            # Use the first game_id in the range for the base folder
            game_id = self.game_id_range[0]
        game_folder = f"game_size{self.game_size}_id{game_id}"
        
        # Combine paths
        organized_path = os.path.join(self.output_path, agent_folder, game_folder)
        return organized_path
    
    def get_game_ids(self) -> List[int]:
        """Get list of game IDs in the specified range."""
        start_id, end_id = self.game_id_range
        return list(range(start_id, end_id + 1))
    
    def get_num_games(self) -> int:
        """Get the number of games in the range."""
        start_id, end_id = self.game_id_range
        return end_id - start_id + 1

# Default debate configurations
def create_default_debate_config(game_size: int = 5, game_id_range: List[int] = None) -> DebateConfig:
    """Create a default debate configuration with 3 agents using flexible naming."""
    # Define the LLM configurations - temperature must be specified for each agent
    llm_configs = [
        {
            "provider": "openai", 
            "model": "gpt-4.1-mini",
        },
        {
            "provider": "gemini",
            "model": "gemini-2.5-flash-lite",
        },
        {
            "provider": "ali",
            "model": "qwen-turbo-latest",
        }
    ]
    
    # Use the flexible configuration to auto-generate agent names
    return create_flexible_debate_config(llm_configs, game_size, game_id_range)

def create_custom_debate_config(agent_configs: List[Dict[str, Any]], game_size: int = 5, game_id_range: List[int] = None) -> DebateConfig:
    """Create a custom debate configuration from a list of agent dictionaries.
    
    Args:
        agent_configs: List of dictionaries with agent configuration
        Example: [
            {"name": "GPT-5", "provider": "openai", "model": "gpt-5-nano"},  # Temperature optional
            {"name": "Gemini", "provider": "gemini", "model": "gemini-2.5-flash", "temperature": 0.2},  # Temperature optional
            {"name": "Qwen", "provider": "ali", "model": "qwen-flash"}  # Temperature optional
        ]
    """
    agents = []
    for config in agent_configs:
        provider = config.get("provider", "openai")
        model = config.get("model", "gpt-5-nano")
        
        # Temperature is optional - use it if provided, otherwise use default
        temperature = config.get("temperature", 0.1)  # Default temperature if not specified
        
        agent = AgentConfig(
            name=config.get("name", f"Agent-{len(agents)+1}"),
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=config.get("max_tokens", 2000),
            reasoning_effort=config.get("reasoning_effort", "medium"),
            verbosity=config.get("verbosity", "low"),
            enabled=config.get("enabled", True)
        )
        agents.append(agent)
    
    return DebateConfig(
        agents=agents,
        max_debate_rounds=config.get("max_debate_rounds", 1),
        enable_self_adjustment=config.get("enable_self_adjustment", True),
        enable_majority_vote=config.get("enable_majority_vote", True),
        supervisor_provider=config.get("supervisor_provider", "openai"),
        supervisor_model=config.get("supervisor_model", "gpt-5-nano"),
        enable_visualization=config.get("enable_visualization", True),
        output_path=config.get("output_path", "./debate_results"),
        save_detailed_logs=config.get("save_detailed_logs", True),
        game_size=game_size,
        game_id_range=game_id_range,
        game_parallel_workers=config.get("game_parallel_workers", 20)
    )

def create_flexible_debate_config(llm_configs: List[Dict[str, Any]], game_size: int = 5, game_id_range: List[int] = None) -> DebateConfig:
    """Create a debate configuration with automatically generated agent names.
    
    Args:
        llm_configs: List of dictionaries with LLM configuration (no need to specify names)
        Example: [
            {"provider": "openai", "model": "gpt-5-nano"},  # Temperature optional
            {"provider": "gemini", "model": "gemini-2.5-flash", "temperature": 0.2},  # Temperature optional
            {"provider": "ali", "model": "qwen-flash"}  # Temperature optional
        ]
    """
    agents = []
    for i, config in enumerate(llm_configs):
        # Generate agent name based on model and provider
        provider = config.get("provider", "openai")
        model = config.get("model", "gpt-5-nano")
        
        # Create a descriptive name
        if provider == "openai":
            agent_name = f"OpenAI-{model}"
        elif provider == "gemini":
            agent_name = f"Gemini-{model}"
        elif provider == "ali":
            agent_name = f"Qwen-{model}"
        elif provider == "cst":
            agent_name = f"CST-{model}"
        else:
            agent_name = f"Agent-{i+1}-{model}"
        
        # Temperature is optional - use it if provided, otherwise use default
        temperature = config.get("temperature", 0.7)  # Default temperature if not specified
        
        agent = AgentConfig(
            name=agent_name,
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=config.get("max_tokens", 2000),
            reasoning_effort=config.get("reasoning_effort", "medium"),
            verbosity=config.get("verbosity", "low"),
            enabled=config.get("enabled", True)
        )
        agents.append(agent)
    
    return DebateConfig(
        agents=agents,
        max_debate_rounds=1,
        enable_self_adjustment=True,
        enable_majority_vote=True,
        supervisor_provider="openai",
        supervisor_model="gpt-5-nano",
        enable_visualization=True,
        output_path="./debate_results",
        save_detailed_logs=True,
        game_size=game_size,
        game_id_range=game_id_range,
        game_parallel_workers=20
    )

