# Multi-Agent Debate System for Knight-Knaves-Spy Games

This system extends the original knight-knaves-spy game solver with a sophisticated multi-agent debate framework, allowing multiple LLM agents to collaborate and debate their solutions to improve accuracy and reasoning quality.

## 🎯 Overview

The multi-agent debate system implements a structured debate process where multiple AI agents:
1. **Initial Proposal**: Each agent independently solves the puzzle
2. **Debate Rounds**: Agents debate each player's role one by one
3. **Self-Adjustment**: Agents can revise their positions based on the debate
4. **Final Vote**: Majority voting determines the final solution
5. **Supervisor Fallback**: GPT-4 makes final decisions when consensus fails

## 🏗️ Architecture

### Core Components

- **`debate_config.py`**: Configuration system for multiple agents
- **`debate_system.py`**: Main debate orchestration system
- **`debate_visualizer.py`**: Performance tracking and visualization
- **`debate_prompts.py`**: Enhanced prompts for debate phases
- **`run_debate.py`**: Command-line interface for running debates
- **`example_debate.py`**: Example scripts and demonstrations

### Key Features

- **Configurable Agents**: Support for OpenAI, Gemini, Ali Cloud, and CST Cloud
- **Parallel Processing**: Efficient parallel execution of agent calls
- **Performance Tracking**: Detailed accuracy and consensus metrics
- **Visualization**: Comprehensive charts and performance matrices
- **Flexible Configuration**: Easy setup for different agent combinations

## 🚀 Quick Start

### 1. Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set up API keys in secret.json
# Format: [{"API provider": "OpenAI", "API key": "your-key"}, ...]
```

### 2. Run Example Debate

```bash
# Run a simple example with 3 agents
python example_debate.py

# Run comparison experiment
python example_debate.py comparison
```

### 3. Run Custom Debates

```bash
# Use predefined configurations
python run_debate.py run default 5 3
python run_debate.py run two_agents 5 5
python run_debate.py run four_agents 5 3

# List available configurations
python run_debate.py list-configs
```

### 4. Custom Agent Configuration

```python
from debate_config import create_custom_debate_config
from debate_system import MultiAgentDebateSystem

# Define your agents
agent_configs = [
    {
        "name": "GPT-5-Analyst",
        "provider": "openai", 
        "model": "gpt-5-nano",
        "temperature": 0.7,
        "reasoning_effort": "high"
    },
    {
        "name": "Gemini-Logician",
        "provider": "gemini",
        "model": "gemini-2.5-flash-lite", 
        "temperature": 0.8
    },
    {
        "name": "Qwen-Strategist",
        "provider": "ali",
        "model": "qwen-flash",
        "temperature": 0.6
    }
]

# Create configuration
debate_config = create_custom_debate_config(agent_configs)

# Run debate
debate_system = MultiAgentDebateSystem(debate_config)
sessions = debate_system.run_batch_debate(games)
```

## 📊 Debate Process

### Phase 1: Initial Proposals
Each agent independently analyzes the game and provides their initial solution.

### Phase 2: Debate Rounds
For each player's role:
1. **Debate Period**: Agents see others' proposals and debate the specific role
2. **Self-Adjustment**: Agents can revise their positions based on the debate
3. **Consensus Check**: System checks if majority agreement is reached

### Phase 3: Final Resolution
1. **Majority Vote**: Collect final votes from all agents
2. **Supervisor Decision**: GPT-4 makes final call if no consensus

## 📈 Performance Tracking

The system tracks:
- **Accuracy by Agent**: Individual agent performance over time
- **Consensus Evolution**: How agreement develops through rounds
- **Performance Improvement**: Changes from initial to final solutions
- **Round-by-Round Analysis**: Detailed tracking of each debate phase

### Visualizations Generated

- **Performance Matrix**: Heatmap of accuracy by agent and round
- **Consensus Tracking**: Line plots showing agreement evolution
- **Agent Comparison**: Box plots and improvement metrics
- **Debate Flow Diagrams**: Visual representation of debate process
- **Summary Reports**: Comprehensive performance analysis

## 🔧 Configuration Options

### Agent Configuration

```python
@dataclass
class AgentConfig:
    name: str                    # Agent identifier
    provider: str               # "openai", "gemini", "ali", "cst"
    model: str                  # Model name
    temperature: float = 0.7    # Response randomness
    max_tokens: int = 2000      # Maximum response length
    reasoning_effort: str = "medium"  # For OpenAI models
    verbosity: str = "medium"   # For OpenAI models
    enabled: bool = True        # Whether to include this agent
```

### Debate Configuration

```python
@dataclass
class DebateConfig:
    agents: List[AgentConfig]           # List of participating agents
    max_debate_rounds: int = 3         # Maximum rounds per player
    enable_self_adjustment: bool = True # Allow position revision
    enable_majority_vote: bool = True   # Use majority voting
    supervisor_provider: str = "openai" # Supervisor LLM provider
    supervisor_model: str = "gpt-4"     # Supervisor model
    enable_visualization: bool = True   # Generate visualizations
    output_path: str = "./debate_results" # Output directory
    save_detailed_logs: bool = True     # Save detailed logs
```

## 📁 Output Structure

```
debate_results/
├── debate_session_1_20241201_143022.json  # Session data
├── performance_matrix_20241201_143022.png # Performance heatmap
├── consensus_tracking_20241201_143022.png # Consensus evolution
├── agent_comparison_20241201_143022.png   # Agent comparison
├── debate_flow_1_20241201_143022.png      # Debate flow diagram
└── summary_report_20241201_143022.txt     # Summary report
```

## 🎯 Example Results

### Sample Debate Session

```
🎯 Game 1 - Final Results:
Ground Truth: {'Alice': 'knave', 'Bob': 'knight', 'Charlie': 'spy'}
Majority Vote: {'Alice': 'knave', 'Bob': 'knight', 'Charlie': 'spy'}
Accuracy: 100.00%

📋 Round-by-Round Analysis:

--- Round 1: Alice ---
Consensus Reached: True
Majority Role: knave
Agent Responses:
  GPT-5-Analyst: knave
    Reasoning: Alice claims to be a knight, but if she were telling the truth...
  Gemini-Logician: knave
    Reasoning: Based on the logical analysis of Alice's statement...
  Qwen-Strategist: knave
    Reasoning: Alice's statement creates a logical contradiction...
```

## 🔬 Research Applications

This system is designed for:
- **Multi-Agent Reasoning**: Studying how multiple AI agents collaborate
- **Debate Effectiveness**: Measuring the impact of structured debate
- **Consensus Building**: Understanding how AI agents reach agreement
- **Performance Improvement**: Comparing single vs multi-agent approaches
- **LLM Evaluation**: Testing reasoning capabilities across different models

## 🛠️ Advanced Usage

### Custom Debate Phases

You can extend the system by modifying the debate phases in `debate_system.py`:

```python
def _run_debate_round(self, game, player_name, round_num, ...):
    # Step 1: Debate period
    debate_responses = self._conduct_debate_for_player(...)
    
    # Step 2: Self-adjustment  
    adjusted_responses = self._conduct_self_adjustment(...)
    
    # Step 3: Custom phase (add your own)
    custom_responses = self._conduct_custom_phase(...)
    
    return DebateRound(...)
```

### Custom Visualizations

Extend `DebateVisualizer` to add new visualization types:

```python
def create_custom_visualization(self, sessions):
    # Your custom visualization code
    pass
```

## 📚 Dependencies

- `matplotlib`: For creating visualizations
- `seaborn`: For enhanced plotting
- `pandas`: For data manipulation
- `numpy`: For numerical operations
- `openai`: For OpenAI API access
- `google-genai`: For Gemini API access

## 🤝 Contributing

To extend the system:

1. **Add New Providers**: Implement new LLM clients in `utility.py`
2. **Custom Debate Phases**: Modify `debate_system.py` for new debate structures
3. **Enhanced Prompts**: Add new prompt templates in `debate_prompts.py`
4. **New Visualizations**: Extend `DebateVisualizer` with additional charts
5. **Performance Metrics**: Add new tracking in `debate_system.py`

## 📄 License

This project extends the original ControlMAD system for multi-agent debate research.
