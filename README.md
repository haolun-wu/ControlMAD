# KKS-bench: Multi-Agent Debate System for Knight-Knave-Spy Games

## 🚀 Quick Start

### Basic Usage
```bash
# Run a single debate session (5 players, 1 game)
python run_debate.py run game_size=5 game_id_range=1,1

# Run multiple games (5 players, games 1-20)
python run_debate.py run game_size=5 game_id_range=1,20

# Run with parallel processing (default: 20 workers)
python run_debate.py run game_size=5 game_id_range=1,10 use_parallel=true game_parallel_workers=20
```

### Advanced Options
```bash
# Enable confidence scoring (0-100) for all model outputs (default: enabled)
python run_debate.py run game_size=5 game_id_range=1,20 self_reported_confidence=true

# Disable confidence scoring
python run_debate.py run game_size=5 game_id_range=1,20 self_reported_confidence=false

# One can directly edit agents in debate/debate_config.py
```

### Command Line Arguments
- `game_size`: Number of players (4, 5, 6, 7, 8, or 9)
- `game_id_range`: Range of games to run (e.g., `1,20` for games 1-20)
- `self_reported_confidence`: Enable/disable confidence scoring (true/false)
- `use_parallel`: Enable/disable parallel processing (true/false)
- `game_parallel_workers`: Number of parallel workers (default: 20)

## 📊 Output & Results

### Generated Files (per game)
- `debate_*.json`: Complete debate session data
- `debate_log_*.log`: Detailed logging with confidence scores and per-player accuracy
- `detailed_tracking_*.txt`: Performance breakdown by agent and round
- `summary_report_*.txt`: Overall performance summary

### Visualizations
- `performance_matrix_*.png`: Overall accuracy matrix across agents and phases
- `detailed_per_player_accuracy_*.png`: Granular per-player accuracy by round
- `consensus_tracking_*.png`: Consensus formation over time
- `agent_comparison_*.png`: Agent performance comparison
- `debate_flow_*.png`: Individual game debate flow diagram

### Key Features
- ✅ **Confidence Scoring**: Models report confidence (0-100) for all decisions
- ✅ **Confidence Sharing**: Models can see each other's confidence during debates
- ✅ **Detailed Logging**: Per-player, per-round accuracy tracking
- ✅ **Advanced Analytics**: Granular performance visualization
- ✅ **Multi-Phase Debates**: Initial proposals → Debate rounds → Final voting → Supervisor decisions

## Game setup
- Three roles:
    - Knights: always telling the truth.
    - Knaves: always lying.
    - Spy: figure crossed.
- Game manager: provide hints. One fixed hint is the number of spies in the game.
- Goal:
    - Each player provides sentence, and the truthfulness align with their roles. The goal is to deduce the roles of all players.

## Codebase structure
- project_types.py
    - save the custom types of the codebase.
- config.py
    - store the configuration of the codebase:
        - game_config: configuration for the groundtruth generation.
        - test_config: configuration for the tests.
- utility.py
    - store help functions for the whole codebase.
- generator.py
    - generate abstract game instance and render it with texts.
- validator.py
    - validate the abstract game instance to make sure that there is a unque solution.
- groundTruth.py
    - generate the ground truth.
- prompts.py
    - store system prompts and output schema.
- test_baseline.py
    - test for the zero-shot of LLMs.

## Folders
- groundtruth: storing all groundtruth of the game.
    - groundtruth are grouped by game size. the record.json helps to generate unique game id.
- test: storing all test results.

