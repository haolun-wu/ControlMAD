# KKS-bench: Multi-Agent Debate System for Knight-Knave-Spy Games

## 🚀 Quick Start

### Simple Usage
```bash
# Run the debate experiment with customized configuration
./run_exp_setting.sh
```

This will run a multi-agent debate with:
- **3 agents**: GPT-5-nano, Gemini-2.5-flash-lite, and Qwen-turbo-latest
- **5 players** per game
- **Games 1-5**
- **Confidence scoring enabled** (1-5 scale)
- **Automatic visualizations** generated after each debate


### Command Line Arguments
- `game_size`: Number of players (4, 5, 6, 7, 8, or 9)
- `game_id_range`: Range of games to run (e.g., `1,20` for games 1-20)
- `self_reported_confidence`: Enable/disable confidence scoring (true/false)
- `use_parallel`: Enable/disable parallel processing (true/false)
- `game_parallel_workers`: Number of parallel workers (default: 20)

## 🏗️ Debate System Architecture

The multi-agent debate system follows a structured approach where agents collaboratively solve Knight-Knave-Spy puzzles through iterative discussion and refinement.

### Debate Flow Overview

**Phase 1: Initialization**
- Each agent independently analyzes the game and provides their complete solution
- Output: Complete solution for all players + reasoning + confidence (1-5 scale)

**Phase 2: Individual Player Debates**
For each player in the game, the system conducts two sub-phases:

#### Round N - Player X: Debate Phase
- **Input to Agent A**: 
  - Other agents' reasoning from previous phase
  - Other agents' solutions specifically for Player X
- **Output from Agent A**: 
  - Agent A's decision on Player X
  - Debate reasoning explaining their position

#### Round N - Player X: Self-Adjustment Phase  
- **Input to Agent A**:
  - All agents' decisions on Player X after debate
  - Complete debate reasoning from all agents
- **Output from Agent A**:
  - Agent A's complete solution for all players
  - Updated reasoning incorporating debate insights

**Phase 3: Final Discussion and Fresh Voting**
- Agents reconsider the entire debate process
- Each agent makes fresh decisions for all players based on complete debate history
- Majority voting is conducted only on these fresh decisions (not cumulative)

**Phase 4: Supervisor Decision (if needed)**
- If no consensus is reached, a supervisor agent makes the final decision
- Supervisor has access to complete debate history and all agent reasoning

### Example Flow (3 Agents: A, B, C)

```
Initialization:
Agent A outputs complete solution for all players + reasoning + confidence

Round 1 - Player 1:
 - Debate Phase:
   Input to A: B and C's reasoning from initialization + their solution on Player 1
   Output from A: A's decision on Player 1 + debate reasoning
 - Self-Adjustment Phase:
   Input to A: A, B, C's decisions on Player 1 after debate + debate reasoning
   Output from A: A's complete solution for all players + reasoning

Round 2 - Player 2:
 - Debate Phase:
   Input to A: B and C's reasoning from Round 1 self-adjustment + their solution on Player 2
   Output from A: A's decision on Player 2 + debate reasoning
 - Self-Adjustment Phase:
   Input to A: A, B, C's decisions on Player 2 after debate + debate reasoning
   Output from A: A's complete solution for all players + reasoning

... (continues for all players)

Final Discussion:
All agents make fresh decisions based on complete debate history
Majority vote determines final solution
```

### Key Features
- ✅ **No Intermediate Voting**: Only final majority vote, no voting after each player debate
- ✅ **Fresh Decision Making**: Final phase requires agents to reconsider everything
- ✅ **Complete Context**: Agents see full debate history in final discussion
- ✅ **Supervisor Fallback**: Human-like supervisor makes final decision if no consensus
- ✅ **Confidence Tracking**: All decisions include confidence scores (1-5)
- ✅ **Flexible Agent Configuration**: Easy to add/remove agents without code changes

## 📊 Output & Results

### Generated Files (per game)
- `debate_*.json`: Complete debate session data
- `debate_log_*.log`: Detailed logging with confidence scores and per-player accuracy
- `detailed_tracking_*.txt`: Performance breakdown by agent and round
- `summary_report_*.txt`: Overall performance summary

### Visualizations
- `performance_matrix_*.png`: Overall accuracy matrix across agents and phases
- `per_player_prediction_matrix_*.png`: Fine-grained matrix showing each agent's prediction (correct/wrong) for each player at different checkpoints
- `player_centric_analysis_*.png`: Analysis comparing initial vs final prediction accuracy for each player
- `detailed_per_player_accuracy_*.png`: Granular per-player accuracy by round
- `consensus_tracking_*.png`: Consensus formation over time
- `agent_comparison_*.png`: Agent performance comparison

### Key Features
- ✅ **Confidence Scoring**: Models report confidence (1-5) for all decisions
- ✅ **Confidence Sharing**: Models can see each other's confidence during debates
- ✅ **Concise Reasoning**: Explanation text constrained to 200 characters for focused responses
- ✅ **Detailed Logging**: Per-player, per-round accuracy tracking
- ✅ **Automatic Visualizations**: Performance charts generated automatically after each debate
- ✅ **Advanced Analytics**: Granular performance visualization
- ✅ **Multi-Phase Debates**: Initial proposals → Individual player debates → Self-adjustment → Final discussion → Supervisor decisions
- ✅ **Clean Data Parsing**: Robust JSON parsing handles models that add extra text after responses
- ✅ **Flexible Agent Setup**: Automatic agent naming and configuration based on LLM inputs

## Game setup
- Three roles:
    - Knights: always telling the truth.
    - Knaves: always lying.
    - Spy: figure crossed.
- Game manager: provide hints. One fixed hint is the number of spies in the game.
- Goal:
    - Each player provides sentence, and the truthfulness align with their roles. The goal is to deduce the roles of all players.

## Codebase Structure

### Core Files
- **`run_debate.py`**: Main entry point for running debate experiments
- **`run_exp_setting.sh`**: Simple script to run default debate configuration
- **`project_types.py`**: Custom data types for the codebase
- **`utility.py`**: Helper functions and LLM client implementations
- **`config.py`**: Configuration for groundtruth generation and testing

### Game Generation & Validation
- **`generator.py`**: Generate abstract game instances and render them with text
- **`validator.py`**: Validate game instances to ensure unique solutions
- **`groundTruth.py`**: Generate ground truth solutions
- **`prompts.py`**: System prompts and output schemas for baseline testing

### Debate System (`debate/` folder)
- **`debate_system.py`**: Core multi-agent debate logic and orchestration
- **`debate_config.py`**: Agent configuration and flexible setup system
- **`debate_prompts.py`**: Prompts for different debate phases and confidence scoring
- **`debate_visualizer.py`**: Visualization tools for debate results and performance
- **`example_debate.py`**: Example usage of the debate system

### Data & Results
- **`groundtruth/`**: Ground truth game data organized by game size (record.json helps generate unique game IDs)
- **`debate_results/`**: Output directory for debate session results and visualizations
- **`test/`**: Test results and validation data

