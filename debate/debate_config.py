from dataclasses import dataclass, field
from typing import List, Dict, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.project_types import base_config, test_setup

@dataclass
class AgentConfig:
    """Configuration for a single LLM agent in the debate system."""
    name: str
    provider: str  # "openai", "gemini", "ali", "cst"
    model: str
    temperature: float = 0.1
    max_tokens: int = 1000
    reasoning_effort: str = "medium"  # for OpenAI
    verbosity: str = "low"  # for OpenAI
    enabled: bool = True

@dataclass
class DebateConfig:
    """Configuration for the multi-agent debate system."""
    agents: List[AgentConfig]
    max_debate_rounds: int = 1
    depth: int = 1  # Number of debate rounds per player (1=1 round, 2=2 rounds, 3=3 rounds)
    enable_self_adjustment: bool = True
    enable_majority_vote: bool = True
    supervisor_provider: str = "openai"
    supervisor_model: str = "gpt-5-nano"
    enable_visualization: bool = True
    output_path: str = "./debate_results"
    save_detailed_logs: bool = True
    game_size: int = 4
    game_id_range: List[int] = field(default_factory=lambda: [1, 1])  # [start_id, end_id] inclusive
    game_id_list: List[int] = field(default_factory=lambda: None)  # Optional specific game IDs to run
    game_parallel_workers: int = 5  # Number of workers for game-level parallel processing
    self_reported_confidence: bool = False  # SRC: Whether models output confidence scores (1-10)
    script_name: str = None  # Name of the script that launched the debate (e.g., "control_hom")
    debate_order_control: int = 0  # 0=use ground truth order, 1=agents decide order after initialization
    
    def __post_init__(self):
        """Set default game_id_range if not provided."""
        # game_id_range now defaults to [1, 1] via field(default_factory)
        pass
    
    def get_organized_output_path(self, game_id: int = None) -> str:
        """Generate organized folder structure based on agent configuration and game parameters."""
        # Create agent folder name: agent3_{model1}_{model2}_{model3}_conf_{true/false}
        agent_count = len(self.agents)
        model_names = [agent.model for agent in self.agents]
        conf_suffix = "true" if self.self_reported_confidence else "false"
        agent_folder = f"agent{agent_count}_{'_'.join(model_names)}_conf_{conf_suffix}"
        
        # Add script name prefix if provided
        if self.script_name:
            agent_folder = f"{self.script_name}_{agent_folder}"
        
        # Create game folder name: game_size{size}_id{num}
        if game_id is None:
            # Use the first game_id in the range for the base folder
            game_id = self.game_id_range[0]
        game_folder = f"game_size{self.game_size}_id{game_id}"
        
        # Combine paths
        organized_path = os.path.join(self.output_path, agent_folder, game_folder)
        return organized_path
    
    def get_game_ids(self) -> List[int]:
        """Get list of game IDs based on game_id_list or game_id_range."""
        if self.game_id_list is not None:
            return sorted(self.game_id_list)  # Return sorted list of specific game IDs
        else:
            # Use game_id_range if game_id_list is not specified
            start_id, end_id = self.game_id_range
            return list(range(start_id, end_id + 1))
    
    def get_num_games(self) -> int:
        """Get the number of games based on game_id_list or game_id_range."""
        if self.game_id_list is not None:
            return len(self.game_id_list)
        else:
            # Use game_id_range if game_id_list is not specified
            start_id, end_id = self.game_id_range
            return end_id - start_id + 1

# Default debate configurations
def create_default_debate_config(game_size: int = 5, game_id_range: List[int] = None, game_id_list: List[int] = None) -> DebateConfig:
    """Create a default debate configuration with 3 agents using flexible naming."""
    # Define the LLM configurations - temperature must be specified for each agent
    llm_configs = [
        {
            "provider": "openai", 
            "model": "gpt-5-nano",
            "reasoning_effort": "low",
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
    return create_flexible_debate_config(llm_configs, game_size, game_id_range, None, 0, 1, game_id_list)

def create_custom_debate_config(agent_configs: List[Dict[str, Any]], game_size: int = 5, game_id_range: List[int] = None, depth: int = 1, game_id_list: List[int] = None) -> DebateConfig:
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
        depth=depth,
        enable_self_adjustment=config.get("enable_self_adjustment", True),
        enable_majority_vote=config.get("enable_majority_vote", True),
        supervisor_provider=config.get("supervisor_provider", "openai"),
        supervisor_model=config.get("supervisor_model", "gpt-5-nano"),
        enable_visualization=config.get("enable_visualization", True),
        output_path=config.get("output_path", "./debate_results"),
        save_detailed_logs=config.get("save_detailed_logs", True),
        game_size=game_size,
        game_id_range=game_id_range,
        game_id_list=game_id_list,
        game_parallel_workers=config.get("game_parallel_workers", 5),
        self_reported_confidence=config.get("self_reported_confidence", True)
    )

def create_flexible_debate_config(llm_configs: List[Dict[str, Any]], game_size: int = 5, game_id_range: List[int] = None, script_name: str = None, debate_order_control: int = 0, depth: int = 1, game_id_list: List[int] = None) -> DebateConfig:
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
    # Track model counts to handle duplicate models
    model_counts = {}
    
    # First pass: count how many times each model appears
    for config in llm_configs:
        provider = config.get("provider", "openai")
        model = config.get("model", "gpt-5-nano")
        model_key = f"{provider}-{model}"
        model_counts[model_key] = model_counts.get(model_key, 0) + 1
    
    # Second pass: assign names with appropriate numbering
    model_used_counts = {}
    
    for i, config in enumerate(llm_configs):
        # Generate agent name based on model and provider
        provider = config.get("provider", "openai")
        model = config.get("model", "gpt-5-nano")
        
        # Create a descriptive name
        if provider == "openai":
            base_name = f"OpenAI-{model}"
        elif provider == "gemini":
            base_name = f"Gemini-{model}"
        elif provider == "ali":
            base_name = f"Qwen-{model}"
        elif provider == "cst":
            base_name = f"CST-{model}"
        else:
            base_name = f"Agent-{i+1}-{model}"
        
        # Track model usage and add number if there are multiple instances
        model_key = f"{provider}-{model}"
        if model_counts[model_key] > 1:
            # Multiple instances of this model, add number starting from 1
            model_used_counts[model_key] = model_used_counts.get(model_key, 0) + 1
            agent_name = f"{base_name}-{model_used_counts[model_key]}"
        else:
            # Only one instance of this model, no number needed
            agent_name = base_name
        
        # Temperature is optional - use it if provided, otherwise use default
        temperature = config.get("temperature", 0.1)  # Default temperature if not specified
        
        agent = AgentConfig(
            name=agent_name,
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=config.get("max_tokens", 1000),
            reasoning_effort=config.get("reasoning_effort", "medium"),
            verbosity=config.get("verbosity", "low"),
            enabled=config.get("enabled", True)
        )
        agents.append(agent)
    
    return DebateConfig(
        agents=agents,
        max_debate_rounds=1,
        depth=depth,
        enable_self_adjustment=True,
        enable_majority_vote=True,
        supervisor_provider="openai",
        supervisor_model="gpt-5-nano",
        enable_visualization=True,
        output_path="./debate_results",
        save_detailed_logs=True,
        game_size=game_size,
        game_id_range=game_id_range,
        game_id_list=game_id_list,
        game_parallel_workers=5,
        self_reported_confidence=True,
        script_name=script_name,
        debate_order_control=debate_order_control
    )

