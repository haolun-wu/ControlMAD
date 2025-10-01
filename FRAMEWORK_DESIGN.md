# ControlMAD: Multi-Agent Debate Framework Design Documentation

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
4. [Chat History Management](#chat-history-management)
5. [Debate Process Flow](#debate-process-flow)
6. [Configuration System](#configuration-system)
7. [Parallel Processing](#parallel-processing)
8. [Visualization & Logging](#visualization--logging)
9. [Design Principles](#design-principles)
10. [Usage Examples](#usage-examples)

## Overview

ControlMAD is a sophisticated multi-agent debate framework designed to evaluate Large Language Model (LLM) reasoning capabilities through structured debates on Knight-Knave-Spy logic puzzles. The framework implements a novel **chat history approach** that provides agents with natural self-awareness and context management through individual conversation histories.

### Key Innovation: Chat History Approach

Unlike traditional multi-agent systems that use monolithic prompts, ControlMAD implements individual chat histories for each agent, enabling:
- **Natural Self-Awareness**: Agents can see their own previous responses in conversation format
- **Better Context Management**: Multi-turn dialogue format instead of single large prompts  
- **Enhanced Memory**: Agents maintain continuity across debate phases
- **Realistic Interaction**: Mimics natural human conversation patterns

## System Architecture

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
        │
┌───────▼──────────────────────────────────────────────────────┐
│                    Agent Clients                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │ OpenAI   │ │ Gemini   │ │ Qwen     │ │ Custom   │       │
│  │ Client   │ │ Client   │ │ Client   │ │ Client   │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
└──────────────────────────────────────────────────────────────┘
        │
┌───────▼──────────────────────────────────────────────────────┐
│                Support Systems                               │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │
│  │ Logging &   │ │ Visualization│ │ Ground Truth           │ │
│  │ HTML Output │ │ System      │ │ Management             │ │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. ChatHistoryDebateSystem

The main orchestrator class that manages the entire debate lifecycle:

```python
class ChatHistoryDebateSystem:
    """
    Enhanced Multi-Agent Debate System with Chat History Support.
    
    This system maintains individual chat histories for each agent, enabling
    better self-awareness through clear separation of agent messages and
    natural multi-turn dialogue format.
    """
```

**Key Responsibilities:**
- Initialize and manage multiple LLM agents
- Coordinate debate phases (Initial → Debate → Self-Adjustment → Final)
- Handle parallel processing of multiple games
- Generate comprehensive logs and visualizations

### 2. AgentChatManager

Manages individual conversation histories for each agent:

```python
class AgentChatManager:
    """Manages chat histories for all agents in a debate session."""
```

**Features:**
- **Individual Chat Histories**: Each agent maintains separate conversation context
- **Message Role Management**: System, user, assistant, other_agent, debate_moderator roles
- **Phase Tracking**: Messages tagged with debate phases (initial, debate, self_adjustment, final)
- **Context Formatting**: Automatic conversion to API-compatible message formats

### 3. Message System

Structured message handling with rich metadata:

```python
@dataclass
class ChatMessage:
    role: MessageRole
    content: str
    timestamp: str
    phase: Optional[str]           # "initial", "debate", "self_adjustment", "final"
    player_focus: Optional[str]    # Which player is being debated
    agent_name: Optional[str]      # Which agent sent this message
    round_number: Optional[int]    # Which debate round
```

### 4. Configuration System

Flexible agent and debate configuration:

```python
@dataclass
class DebateConfig:
    agents: List[AgentConfig]
    depth: int = 1                    # Number of debate rounds per player
    enable_self_adjustment: bool = True
    self_reported_confidence: bool = False
    debate_order_control: int = 0     # 0=ground truth order, 1=agent-decided
    game_parallel_workers: int = 20
```

## Chat History Management

### Message Flow Architecture

```
Agent Perspective (e.g., OpenAI-gpt-5-nano):

┌─────────────────────────────────────────────────────────────┐
│                    Chat History                             │
├─────────────────────────────────────────────────────────────┤
│ [SYSTEM] You are OpenAI-gpt-5-nano. You are participating  │
│          in a multi-agent debate about a Knight-Knave-Spy  │
│          game...                                            │
├─────────────────────────────────────────────────────────────┤
│ [USER] Game Information:                                    │
│        Player Alice: "I am a knight"                       │
│        Player Bob: "Alice is lying"                        │
│        Hint: There is exactly one spy                      │
├─────────────────────────────────────────────────────────────┤
│ [ASSISTANT] My initial analysis:                            │
│             {"players": [{"name": "Alice", "role": "spy"}]} │
├─────────────────────────────────────────────────────────────┤
│ [OTHER_AGENT] Gemini-gemini-2.5-flash initially proposed:  │
│               Alice=knight, Bob=knave                       │
│               Their reasoning: Alice's statement is true... │
├─────────────────────────────────────────────────────────────┤
│ [DEBATE_MODERATOR] Now debating Alice's role. Consider     │
│                    other agents' positions...              │
├─────────────────────────────────────────────────────────────┤
│ [ASSISTANT] After considering others' views:               │
│             {"role": "knight", "agree_with": ["Gemini"]}   │
└─────────────────────────────────────────────────────────────┘
```

### Benefits of Chat History Approach

1. **Natural Self-Awareness**: Agents see their own responses as `[ASSISTANT]` messages
2. **Context Continuity**: Full conversation history maintained across phases
3. **Peer Awareness**: Other agents' positions formatted as `[OTHER_AGENT]` messages
4. **Structured Interaction**: Clear role separation (system, user, assistant, moderator)

## Debate Process Flow

### Phase Structure

```
Game Session Flow:
┌─────────────────────────────────────────────────────────────┐
│                    Phase 1: Initial Proposals              │
│  • Each agent analyzes game independently                  │
│  • Generates initial role assignments                      │
│  • Optional confidence scoring                             │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                    Phase 2: Debate Rounds                  │
│  For each player (Alice, Bob, Charlie...):                │
│    For depth_round in range(1, depth+1):                  │
│      • Agents see others' positions                       │
│      • Express agreement/disagreement                     │
│      • Provide reasoning                                  │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                Phase 3: Self-Adjustment                    │
│  For each player:                                          │
│    • Agents review all debate rounds                      │
│    • Adjust their positions if convinced                  │
│    • Provide updated reasoning                            │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                 Phase 4: Final Discussion                  │
│  • Agents make final decisions for ALL players            │
│  • Access to complete debate history                      │
│  • Generate final role assignments                        │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                   Consensus & Voting                       │
│  • Check for majority consensus                           │
│  • If no consensus: Supervisor agent decides              │
│  • Generate final session results                         │
└─────────────────────────────────────────────────────────────┘
```

### Depth Control

The `depth` parameter controls debate intensity:

- **depth=1**: Single debate round per player (faster, basic interaction)
- **depth=2**: Two debate rounds per player (more thorough discussion)  
- **depth=3**: Three debate rounds per player (extensive deliberation)

### Example Debate Sequence (depth=2)

```
Player Alice (depth=2):
  Debate Round 1: Initial positions exchange
  Debate Round 2: Refined arguments after seeing Round 1
  Self-Adjustment: Final position after all debate rounds

Player Bob (depth=2):
  Debate Round 1: Positions exchange (with Alice context)
  Debate Round 2: Refined arguments
  Self-Adjustment: Final position

... (continue for all players)
```

## Configuration System

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

### Flexible Configuration Creation

```python
# Method 1: Default configuration (3 agents)
debate_config = create_default_debate_config(game_size=5, game_id_range=[1, 20])

# Method 2: Custom configuration with explicit names
agent_configs = [
    {"name": "GPT-5", "provider": "openai", "model": "gpt-5-nano"},
    {"name": "Gemini", "provider": "gemini", "model": "gemini-2.5-flash"},
]
debate_config = create_custom_debate_config(agent_configs, game_size=5)

# Method 3: Flexible configuration (auto-generated names)
llm_configs = [
    {"provider": "openai", "model": "gpt-5-nano"},
    {"provider": "gemini", "model": "gemini-2.5-flash", "temperature": 0.2},
    {"provider": "ali", "model": "qwen-turbo-latest"},
]
debate_config = create_flexible_debate_config(llm_configs, game_size=5)
```

### Advanced Configuration Options

```python
@dataclass
class DebateConfig:
    # Core debate parameters
    depth: int = 1                          # Debate rounds per player
    enable_self_adjustment: bool = True     # Allow position changes
    enable_majority_vote: bool = True       # Use majority for consensus
    
    # Performance & scaling
    game_parallel_workers: int = 20         # Parallel game processing
    
    # Advanced features
    self_reported_confidence: bool = False  # Confidence scoring (1-10)
    debate_order_control: int = 0          # 0=ground truth, 1=agent-decided
    
    # Output & organization
    output_path: str = "./debate_results"
    script_name: str = None                # For organized folder naming
```

## Parallel Processing

### Multi-Level Parallelization

```python
class ParallelProcessor:
    """
    Utility class for parallel processing of tasks using ThreadPoolExecutor.
    Supports configurable number of workers and maintains order of results.
    """
```

**Parallelization Levels:**

1. **Game-Level Parallelism**: Multiple games processed simultaneously
   ```python
   # Process 20 games in parallel with 20 workers
   debate_config.game_parallel_workers = 20
   sessions = debate_system.run_parallel_batch_debate(games)
   ```

2. **Agent-Level Parallelism**: Agent responses within each phase
   ```python
   # All agents respond simultaneously in each debate phase
   responses = self._get_parallel_agent_responses(agents, prompt)
   ```

3. **Visualization Parallelism**: Chart generation across multiple sessions
   ```python
   # Generate visualizations for multiple sessions in parallel
   with Pool(processes=num_processes) as pool:
       results = pool.map(process_single_session, args_list)
   ```

### Performance Benefits

- **Scalability**: Handle hundreds of games efficiently
- **Resource Utilization**: Maximize CPU and API throughput
- **Fault Tolerance**: Individual failures don't stop entire batch
- **Progress Tracking**: Real-time completion monitoring

## Visualization & Logging

### Comprehensive Logging System

```python
# Multi-format logging
self._setup_logging(game_id, create_log_file=True)

# Outputs:
# - Console: Real-time progress
# - .log file: Structured text logs  
# - .html file: Rich formatted logs with styling
# - .json files: Session data for analysis
```

### Rich Visualization Suite

```python
class ChatHistoryDebateVisualizer:
    """Visualization system for chat history multi-agent debate performance."""
```

**Generated Visualizations:**

1. **Performance Matrix**: Overall accuracy heatmaps
2. **Per-Player Prediction Matrix**: Fine-grained role prediction accuracy
3. **Player-Centric Analysis**: Initial vs Final position changes
4. **Consensus Tracking**: Agreement patterns across rounds
5. **Agent Comparison**: Individual agent performance metrics
6. **Detailed Tracking Reports**: Comprehensive analysis documents

### Organized Output Structure

```
debate_results/
├── agent3_gpt-5-nano_gemini-2.5-flash_qwen-turbo_conf_false/
│   ├── game_size6_id1/
│   │   ├── debate_log_game1_20250930_180022.html
│   │   ├── debate_log_game1_20250930_180022.log  
│   │   ├── debate_session_game1_20250930_180022.json
│   │   ├── chat_history_OpenAI-gpt-5-nano_game1_20250930_180022.json
│   │   ├── chat_history_Gemini-gemini-2.5-flash_game1_20250930_180022.json
│   │   └── performance_matrix_1.png
│   └── game_size6_id2/
│       └── ... (similar structure)
└── script_name_agent3_model1_model2_model3_conf_true/
    └── ... (organized by script name)
```

## Design Principles

### 1. **Natural Agent Interaction**
- Chat history mimics human conversation patterns
- Clear role separation (system, user, assistant, other agents)
- Temporal message ordering with metadata

### 2. **Modular Architecture**
- Separate concerns: chat management, debate logic, visualization
- Pluggable agent providers (OpenAI, Gemini, Qwen, custom)
- Configurable debate parameters and phases

### 3. **Scalable Performance**
- Multi-level parallelization (games, agents, visualizations)
- Efficient resource utilization
- Fault-tolerant processing

### 4. **Comprehensive Analysis**
- Rich logging with multiple output formats
- Detailed performance visualizations
- Structured data export for further analysis

### 5. **Research-Oriented Design**
- Controlled experimental parameters (depth, confidence, order)
- Reproducible results with detailed logging
- Flexible configuration for different research questions

## Usage Examples

### Basic Usage

```bash
# Run default 3-agent debate on 6-player games
python run_debate_chat.py run game_size=6 game_id_range=1,20

# Enable confidence scoring and deeper debates
python run_debate_chat.py run game_size=6 game_id_range=1,20 \
    self_reported_confidence=true depth=2
```

### Advanced Configuration

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

---

This framework represents a significant advancement in multi-agent debate systems, providing natural agent interaction through chat histories while maintaining the scalability and analytical depth required for serious LLM reasoning research.
