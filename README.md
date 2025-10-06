# ControlMAD: Multi-Agent Debate System for Knight-Knave-Spy Games

A sophisticated research framework for evaluating Large Language Model (LLM) reasoning capabilities through structured multi-agent debates on Knight-Knave-Spy logic puzzles. The framework implements a novel **chat history approach** that provides agents with natural self-awareness and context management through individual conversation histories.

## 🎯 Key Innovation: Chat History Approach

Unlike traditional multi-agent systems that use monolithic prompts, ControlMAD implements individual chat histories for each agent, enabling:
- **Natural Self-Awareness**: Agents can see their own previous responses in conversation format
- **Better Context Management**: Multi-turn dialogue format instead of single large prompts  
- **Enhanced Memory**: Agents maintain continuity across debate phases
- **Realistic Interaction**: Mimics natural human conversation patterns

## 📊 Available Datasets

The framework includes comprehensive ground truth datasets in the `groundtruth/` folder:

| Game Size | Total Games | Files |
|-----------|-------------|-------|
| **4-player** | 300 games | `4.jsonl`, `4_readable.md` |
| **5-player** | 300 games | `5.jsonl`, `5_readable.md` |
| **6-player** | 300 games | `6.jsonl`, `6_readable.md` |
| **7-player** | 300 games | `7.jsonl`, `7_readable.md` |
| **8-player** | 300 games | `8.jsonl`, `8_readable.md` |
| **9-player** | 300 games | `9.jsonl`, `9_readable.md` |

Each dataset includes:
- **JSONL format**: Machine-readable game data for automated processing
- **Readable format**: Human-readable markdown files for easy inspection
- **Total coverage**: 1,800 Knight-Knave-Spy logic puzzles across all game sizes

## 🚀 Quick Start

### Setup
1. Create `secret.json` with your API keys in the root directory:
```json
{
  "openai_api_key": "your_openai_key",
  "gemini_api_key": "your_gemini_key", 
  "ali_api_key": "your_ali_key"
}
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Basic Usage

#### Run Heterogeneous Agent Mixes (Recommended Examples)

The framework includes four pre-configured heterogeneous agent mixes that demonstrate different model combinations:

**Mix A - High Performance Mix:**
```bash
./run_sh/run_het_mix_A.sh
```
- GPT-5 Nano (medium reasoning effort)
- Qwen Turbo Latest
- Gemini 2.5 Flash Lite

**Mix B - Balanced Performance Mix:**
```bash
./run_sh/run_het_mix_B.sh
```
- GPT-5 Mini (low reasoning effort)
- Qwen Flash
- Gemini 2.5 Flash Lite

**Mix C - Alternative Provider Mix:**
```bash
./run_sh/run_het_mix_C.sh
```
- Gemini 2.5 Flash
- Qwen Flash
- GPT-4.1 Mini

**Mix D - OpenAI Family Mix:**
```bash
./run_sh/run_het_mix_D.sh
```
- GPT-5 Nano (medium reasoning effort)
- GPT-5 Mini (low reasoning effort)
- Gemini 2.5 Flash

Each mix runs on game sizes 4, 6, and 8 with 100 games per size.

#### Control Experiments

**Confidence Control:**
```bash
./run_sh/control_conf.sh
```

**Depth Control:**
```bash
./run_sh/control_depth.sh
```

**Homogeneous Strong Models:**
```bash
./run_sh/control_hom_strong.sh
```

**Homogeneous Weak Models:**
```bash
./run_sh/control_hom_weak.sh
```

**Number of Agents Control:**
```bash
./run_sh/control_num.sh
```

**Debate Order Control:**
```bash
./run_sh/control_order.sh
```

#### Visualize Results
```bash
./run_sh/visualize_all_results.sh
```

## 🛠️ Advanced Usage

### Command Line Interface

#### Flexible Configuration
```bash
# Custom agent setup with flexible configuration
python run_debate_chat.py flexible \
    llm_configs='[
        {"provider":"openai","model":"gpt-5-nano"},
        {"provider":"gemini","model":"gemini-2.5-flash","temperature":0.2},
        {"provider":"ali","model":"qwen-turbo-latest"}
    ]' \
    game_size=6 game_id_range=1,50 \
    depth=3 self_reported_confidence=true \
    script_name=experiment_1
```

#### Basic Configuration
```bash
# Run default 3-agent debate on 6-player games
python run_debate_chat.py run game_size=6 game_id_range=1,20

# Enable confidence scoring and deeper debates
python run_debate_chat.py run game_size=6 game_id_range=1,20 \
    self_reported_confidence=true depth=2
```

### Programmatic Usage

```python
from debate.debate_system_chat import ChatHistoryDebateSystem
from debate.debate_config import create_flexible_debate_config

# Create configuration
llm_configs = [
    {"provider": "openai", "model": "gpt-5-nano"},
    {"provider": "gemini", "model": "gemini-2.5-flash"},
]
config = create_flexible_debate_config(llm_configs, game_size=6, depth=2)

# Initialize system
debate_system = ChatHistoryDebateSystem(config)

# Run debates
games = load_ground_truth_games(6, [1, 10])
sessions = debate_system.run_parallel_batch_debate(games)

# Generate visualizations
from debate.debate_visualizer import ChatHistoryDebateVisualizer
visualizer = ChatHistoryDebateVisualizer()
results = visualizer.create_all_visualizations(sessions)
```

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ControlMAD Framework                     │
├─────────────────────────────────────────────────────────────┤
│  Entry Point: run_debate_chat.py                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              ChatHistoryDebateSystem                        │
│  • Orchestrates entire debate process                      │
│  • Manages agent lifecycle                                 │
│  • Coordinates parallel processing                         │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
┌───────▼──────┐ ┌───▼────┐ ┌──────▼──────┐
│ AgentChat    │ │ Debate │ │ Parallel    │
│ Manager      │ │ Config │ │ Processor   │
│              │ │        │ │             │
│ • Individual │ │ • Agent│ │ • Game-level│
│   chat       │ │   setup│ │   parallel  │
│   histories  │ │ • Debate│ │   processing│
│ • Message    │ │   params│ │ • Thread    │
│   management │ │        │ │   management│
└──────────────┘ └────────┘ └─────────────┘
```

## 🔄 Debate Process Flow

### Phase Structure

1. **Initial Proposals**: Each agent analyzes game independently
2. **Debate Rounds**: For each player, agents exchange positions and reasoning
3. **Self-Adjustment**: Agents review all debate rounds and adjust positions
4. **Final Discussion**: Agents make final decisions with complete context
5. **Consensus & Voting**: Check for majority consensus or supervisor decision

### Depth Control

- **depth=1**: Single debate round per player (faster, basic interaction)
- **depth=2**: Two debate rounds per player (more thorough discussion)  
- **depth=3**: Three debate rounds per player (extensive deliberation)

## 📈 Results & Visualization

### Output Structure
```
debate_results/
├── mix_A_agent3_gpt-5-nano_qwen-turbo-latest_gemini-2.5-flash-lite_conf_false/
│   ├── game_size4_id1/
│   │   ├── debate_log_game1_20250930_180022.html
│   │   ├── debate_log_game1_20250930_180022.log  
│   │   ├── debate_session_game1_20250930_180022.json
│   │   ├── chat_history_OpenAI-gpt-5-nano_game1_20250930_180022.json
│   │   └── performance_matrix_1.png
│   └── game_size6_id1/
│       └── ... (similar structure)
```

### Generated Visualizations

1. **Performance Matrix**: Overall accuracy heatmaps
2. **Per-Player Prediction Matrix**: Fine-grained role prediction accuracy
3. **Player-Centric Analysis**: Initial vs Final position changes
4. **Consensus Tracking**: Agreement patterns across rounds
5. **Agent Comparison**: Individual agent performance metrics
6. **Detailed Tracking Reports**: Comprehensive analysis documents

## ⚙️ Configuration Options

### Agent Configuration
```python
@dataclass
class AgentConfig:
    name: str                    # Auto-generated: "OpenAI-gpt-5-nano"
    provider: str               # "openai", "gemini", "ali", "cst"
    model: str                  # "gpt-5-nano", "gemini-2.5-flash"
    temperature: float = 0.1    # Optional, defaults to 0.1
    max_tokens: int = 1000
    reasoning_effort: str = "medium"  # OpenAI specific
    enabled: bool = True
```

### Debate Configuration
```python
@dataclass
class DebateConfig:
    depth: int = 1                          # Debate rounds per player
    enable_self_adjustment: bool = True     # Allow position changes
    enable_majority_vote: bool = True       # Use majority for consensus
    game_parallel_workers: int = 20         # Parallel game processing
    self_reported_confidence: bool = False  # Confidence scoring (1-10)
    debate_order_control: int = 0          # 0=ground truth, 1=agent-decided
    output_path: str = "./debate_results"
    script_name: str = None                # For organized folder naming
```

## 🔧 Design Principles

1. **Natural Agent Interaction**: Chat history mimics human conversation patterns
2. **Modular Architecture**: Separate concerns with pluggable agent providers
3. **Scalable Performance**: Multi-level parallelization for efficiency
4. **Comprehensive Analysis**: Rich logging with multiple output formats
5. **Research-Oriented Design**: Controlled experimental parameters for reproducibility

## 📚 Research Applications

This framework is designed for serious LLM reasoning research, providing:
- Controlled experimental parameters (depth, confidence, order)
- Reproducible results with detailed logging
- Flexible configuration for different research questions
- Comprehensive performance analysis tools

The chat history approach represents a significant advancement in multi-agent debate systems, enabling more natural and effective agent interactions while maintaining the scalability and analytical depth required for rigorous LLM evaluation.