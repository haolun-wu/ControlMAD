#!/usr/bin/env python3
"""
Multi-Agent Debate System for Knight-Knaves-Spy Games

This script runs the complete debate pipeline with configurable agents.
"""

import os
import sys
import json
from typing import List, Dict, Any

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from project_types import ground_truth
from debate_config import DebateConfig, create_default_debate_config, create_custom_debate_config
from debate_system import MultiAgentDebateSystem
from debate_visualizer import DebateVisualizer
from test_baseline import Test
from config import test_config

def load_ground_truth_games(game_size: int = 5, num_games: int = 10) -> List[ground_truth]:
    """Load ground truth games for testing."""
    ground_truth_list = []
    
    try:
        ground_truth_path = f"./groundtruth/{game_size}.jsonl"
        with open(ground_truth_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                if line_num > num_games:  # Limit number of games
                    break
                    
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    gt = ground_truth(
                        game_id=data['game_id'],
                        num_player=data['num_player'],
                        num_spy=data['num_spy'],
                        num_hint=data['num_hint'],
                        raw_schema=data['raw_schema'],
                        raw_statement=data['raw_statement'],
                        raw_solution=data['raw_solution'],
                        text_game=data['text_game'],
                        text_solution=data['text_solution']
                    )
                    ground_truth_list.append(gt)
                    
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {line_num}: {e}")
                    continue
                except KeyError as e:
                    print(f"Warning: Missing required field {e} on line {line_num}")
                    continue
                    
    except FileNotFoundError:
        print(f"Error: Ground truth file not found: {ground_truth_path}")
        print("Please run groundTruth.py first to generate test data.")
        return []
    except Exception as e:
        print(f"Error reading ground truth file: {e}")
        return []
    
    return ground_truth_list

def create_example_configs() -> Dict[str, DebateConfig]:
    """Create example debate configurations."""
    
    configs = {}
    
    # Default 3-agent configuration
    configs['default'] = create_default_debate_config()
    
    # 2-agent configuration
    configs['two_agents'] = create_custom_debate_config([
        {"name": "GPT-5", "provider": "openai", "model": "gpt-5-nano"},
        {"name": "Gemini", "provider": "gemini", "model": "gemini-2.5-flash-lite"}
    ])
    
    # 4-agent configuration
    configs['four_agents'] = create_custom_debate_config([
        {"name": "GPT-5", "provider": "openai", "model": "gpt-5-nano"},
        {"name": "Gemini", "provider": "gemini", "model": "gemini-2.5-flash-lite"},
        {"name": "Qwen", "provider": "ali", "model": "qwen-flash"},
        {"name": "CST", "provider": "cst", "model": "gpt-oss-120b"}
    ])
    
    # All OpenAI agents with different models
    configs['openai_variants'] = create_custom_debate_config([
        {"name": "GPT-5-Nano", "provider": "openai", "model": "gpt-5-nano", "temperature": 0.7},
        {"name": "GPT-5", "provider": "openai", "model": "gpt-5", "temperature": 0.8},
        {"name": "GPT-4o", "provider": "openai", "model": "gpt-4o", "temperature": 0.6}
    ])
    
    return configs

def run_single_debate_session(config_name: str = "default", 
                            game_size: int = 5, 
                            num_games: int = 3,
                            enable_visualization: bool = True):
    """Run a single debate session with specified configuration."""
    
    print("🚀 Starting Multi-Agent Debate System")
    print("=" * 50)
    
    # Load configuration
    configs = create_example_configs()
    if config_name not in configs:
        print(f"Error: Configuration '{config_name}' not found.")
        print(f"Available configurations: {list(configs.keys())}")
        return
    
    debate_config = configs[config_name]
    print(f"📋 Using configuration: {config_name}")
    print(f"🤖 Agents: {[agent.name for agent in debate_config.agents]}")
    
    # Load ground truth games
    print(f"\n📚 Loading {num_games} games of size {game_size}...")
    games = load_ground_truth_games(game_size, num_games)
    
    if not games:
        print("❌ No games loaded. Please generate ground truth data first.")
        return
    
    print(f"✅ Loaded {len(games)} games")
    
    # Initialize debate system
    print(f"\n🔧 Initializing debate system...")
    debate_system = MultiAgentDebateSystem(debate_config)
    
    # Run debate sessions
    print(f"\n🎯 Running debate sessions...")
    sessions = debate_system.run_batch_debate(games)
    
    # Create visualizations
    if enable_visualization:
        print(f"\n🎨 Creating visualizations...")
        visualizer = DebateVisualizer(debate_config.output_path)
        visualization_paths = visualizer.create_all_visualizations(sessions)
        
        print(f"\n📊 Visualization files created:")
        for name, path in visualization_paths.items():
            print(f"  - {name}: {path}")
    
    # Print summary
    print(f"\n📈 DEBATE SUMMARY")
    print("=" * 30)
    
    total_sessions = len(sessions)
    consensus_reached = sum(1 for s in sessions if s.final_vote)
    supervisor_used = sum(1 for s in sessions if s.supervisor_decision)
    
    print(f"Total Sessions: {total_sessions}")
    print(f"Consensus Reached: {consensus_reached}/{total_sessions} ({consensus_reached/total_sessions*100:.1f}%)")
    print(f"Supervisor Used: {supervisor_used}/{total_sessions} ({supervisor_used/total_sessions*100:.1f}%)")
    
    # Calculate overall accuracy
    all_accuracies = []
    for session in sessions:
        if session.final_vote:
            correct = 0
            total = len(session.ground_truth_solution)
            for player, role in session.ground_truth_solution.items():
                if session.final_vote.get(player) == role:
                    correct += 1
            accuracy = correct / total if total > 0 else 0
            all_accuracies.append(accuracy)
    
    if all_accuracies:
        import numpy as np
        print(f"Average Final Accuracy: {np.mean(all_accuracies):.3f}")
        print(f"Accuracy Range: {np.min(all_accuracies):.3f} - {np.max(all_accuracies):.3f}")
    
    print(f"\n✅ Debate system completed successfully!")
    print(f"📁 Results saved to: {debate_config.output_path}")

def run_custom_debate(agent_configs: List[Dict[str, Any]], 
                     game_size: int = 5, 
                     num_games: int = 3,
                     enable_visualization: bool = True):
    """Run debate with custom agent configuration."""
    
    print("🚀 Starting Custom Multi-Agent Debate System")
    print("=" * 50)
    
    # Create custom configuration
    debate_config = create_custom_debate_config(agent_configs)
    print(f"🤖 Custom agents: {[agent.name for agent in debate_config.agents]}")
    
    # Load ground truth games
    print(f"\n📚 Loading {num_games} games of size {game_size}...")
    games = load_ground_truth_games(game_size, num_games)
    
    if not games:
        print("❌ No games loaded. Please generate ground truth data first.")
        return
    
    print(f"✅ Loaded {len(games)} games")
    
    # Initialize debate system
    print(f"\n🔧 Initializing debate system...")
    debate_system = MultiAgentDebateSystem(debate_config)
    
    # Run debate sessions
    print(f"\n🎯 Running debate sessions...")
    sessions = debate_system.run_batch_debate(games)
    
    # Create visualizations
    if enable_visualization:
        print(f"\n🎨 Creating visualizations...")
        visualizer = DebateVisualizer(debate_config.output_path)
        visualization_paths = visualizer.create_all_visualizations(sessions)
        
        print(f"\n📊 Visualization files created:")
        for name, path in visualization_paths.items():
            print(f"  - {name}: {path}")
    
    print(f"\n✅ Custom debate system completed successfully!")
    print(f"📁 Results saved to: {debate_config.output_path}")

def main():
    """Main function with command line interface."""
    
    if len(sys.argv) < 2:
        print("Usage: python run_debate.py <command> [options]")
        print("\nCommands:")
        print("  run <config_name> [game_size] [num_games]")
        print("  custom <agent_configs_json> [game_size] [num_games]")
        print("  list-configs")
        print("\nExamples:")
        print("  python run_debate.py run default 5 3")
        print("  python run_debate.py run two_agents 5 5")
        print("  python run_debate.py list-configs")
        return
    
    command = sys.argv[1]
    
    if command == "run":
        config_name = sys.argv[2] if len(sys.argv) > 2 else "default"
        game_size = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        num_games = int(sys.argv[4]) if len(sys.argv) > 4 else 3
        
        run_single_debate_session(config_name, game_size, num_games)
        
    elif command == "custom":
        if len(sys.argv) < 3:
            print("Error: Please provide agent configurations as JSON string")
            return
        
        try:
            agent_configs = json.loads(sys.argv[2])
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON for agent configurations: {e}")
            return
        
        game_size = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        num_games = int(sys.argv[4]) if len(sys.argv) > 4 else 3
        
        run_custom_debate(agent_configs, game_size, num_games)
        
    elif command == "list-configs":
        configs = create_example_configs()
        print("Available configurations:")
        for name, config in configs.items():
            print(f"\n{name}:")
            print(f"  Agents: {[agent.name for agent in config.agents]}")
            print(f"  Max Rounds: {config.max_debate_rounds}")
            print(f"  Self-Adjustment: {config.enable_self_adjustment}")
            print(f"  Majority Vote: {config.enable_majority_vote}")
    
    else:
        print(f"Error: Unknown command '{command}'")
        print("Use 'python run_debate.py' without arguments to see usage.")

if __name__ == "__main__":
    main()
