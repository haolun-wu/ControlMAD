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

from utils.project_types import ground_truth
from debate.debate_config import DebateConfig, create_default_debate_config, create_custom_debate_config, create_flexible_debate_config
from debate.debate_system import MultiAgentDebateSystem
from utils.config import test_config

def load_ground_truth_games(game_size: int = 5, game_id_range: List[int] = None) -> List[ground_truth]:
    """Load ground truth games for testing."""
    if game_id_range is None:
        game_id_range = [1, 1]  # Default to single game
    
    start_id, end_id = game_id_range
    ground_truth_list = []
    
    try:
        ground_truth_path = f"./groundtruth/{game_size}.jsonl"
        with open(ground_truth_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    game_id = data['game_id']
                    
                    # Filter by actual game_id, not line number
                    if game_id < start_id:  # Skip games before start_id
                        continue
                    if game_id > end_id:  # Stop after end_id
                        break
                    
                    gt = ground_truth(
                        game_id=game_id,
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


def run_debates_with_system(debate_system, games, use_parallel=True):
    """Helper function to run debates with either parallel or sequential processing."""
    print(f"\n🎯 Running debates...")
    if use_parallel:
        print("🔄 Using parallel processing for multiple games")
        return debate_system.run_parallel_batch_debate(games)
    else:
        print("🔄 Using sequential processing for multiple games")
        return debate_system.run_batch_debate(games)




def run_single_debate_session(game_size: int = 5, 
                            game_id_range: List[int] = None,
                            use_parallel: bool = True,
                            game_parallel_workers: int = 20,
                            self_reported_confidence: bool = False):
    """Run a single debate with default configuration."""
    
    print("🚀 Starting Multi-Agent Debate System")
    print("=" * 50)
    
    # Set default game_id_range if not provided
    if game_id_range is None:
        game_id_range = [1, 3]  # Default to games 1-3
    
    # Load default configuration
    debate_config = create_default_debate_config(game_size, game_id_range)
    # Override parameters if specified
    debate_config.game_parallel_workers = game_parallel_workers
    debate_config.self_reported_confidence = self_reported_confidence
    print(f"📋 Using default configuration")
    print(f"🤖 Agents: {[agent.name for agent in debate_config.agents]}")
    print(f"🔧 Game parallel workers: {debate_config.game_parallel_workers}")
    print(f"📊 Self-reported confidence: {'enabled' if self_reported_confidence else 'disabled'}")
    
    # Load ground truth games
    num_games = debate_config.get_num_games()
    print(f"\n📚 Loading games {game_id_range[0]}-{game_id_range[1]} of size {game_size} ({num_games} games)...")
    games = load_ground_truth_games(game_size, game_id_range)
    
    if not games:
        print("❌ No games loaded. Please generate ground truth data first.")
        return
    
    print(f"✅ Loaded {len(games)} games")
    
    # Initialize debate system
    print(f"\n🔧 Initializing debate system...")
    debate_system = MultiAgentDebateSystem(debate_config)
    
    # Run debates
    print(f"\n🎯 Running debates...")
    sessions = run_debates_with_system(debate_system, games, use_parallel)
    
    # Print summary
    print(f"\n📈 DEBATE SUMMARY")
    print("=" * 30)
    
    total_debates = len(sessions)
    consensus_reached = sum(1 for s in sessions if s.final_vote)
    supervisor_used = sum(1 for s in sessions if s.supervisor_decision)
    
    print(f"Total Debates: {total_debates}")
    print(f"Consensus Reached: {consensus_reached}/{total_debates} ({consensus_reached/total_debates*100:.1f}%)")
    print(f"Supervisor Used: {supervisor_used}/{total_debates} ({supervisor_used/total_debates*100:.1f}%)")
    
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
                     game_id_range: List[int] = None,
                     use_parallel: bool = True):
    """Run debate with custom agent configuration."""
    
    print("🚀 Starting Custom Multi-Agent Debate System")
    print("=" * 50)
    
    # Set default game_id_range if not provided
    if game_id_range is None:
        game_id_range = [1, 3]  # Default to games 1-3
    
    # Create custom configuration
    debate_config = create_custom_debate_config(agent_configs, game_size, game_id_range)
    print(f"🤖 Custom agents: {[agent.name for agent in debate_config.agents]}")
    
    # Load ground truth games
    num_games = debate_config.get_num_games()
    print(f"\n📚 Loading games {game_id_range[0]}-{game_id_range[1]} of size {game_size} ({num_games} games)...")
    games = load_ground_truth_games(game_size, game_id_range)
    
    if not games:
        print("❌ No games loaded. Please generate ground truth data first.")
        return
    
    print(f"✅ Loaded {len(games)} games")
    
    # Initialize debate system
    print(f"\n🔧 Initializing debate system...")
    debate_system = MultiAgentDebateSystem(debate_config)
    
    # Run debates
    print(f"\n🎯 Running debates...")
    sessions = run_debates_with_system(debate_system, games, use_parallel)
    
    print(f"\n✅ Custom debate system completed successfully!")
    # Show the base agent folder path
    agent_count = len(debate_config.agents)
    model_names = [agent.model for agent in debate_config.agents]
    conf_suffix = "true" if debate_config.self_reported_confidence else "false"
    agent_folder = f"agent{agent_count}_{'_'.join(model_names)}_conf_{conf_suffix}"
    base_path = os.path.join(debate_config.output_path, agent_folder)
    print(f"📁 Results saved to: {base_path}")

def run_flexible_debate(llm_configs: List[Dict[str, Any]], 
                       game_size: int = 5, 
                       game_id_range: List[int] = None,
                       use_parallel: bool = True):
    """Run debate with flexible agent configuration (auto-generated names)."""
    
    print("🚀 Starting Flexible Multi-Agent Debate System")
    print("=" * 50)
    
    # Set default game_id_range if not provided
    if game_id_range is None:
        game_id_range = [1, 3]  # Default to games 1-3
    
    # Create flexible configuration
    debate_config = create_flexible_debate_config(llm_configs, game_size, game_id_range)
    print(f"🤖 Auto-generated agents: {[agent.name for agent in debate_config.agents]}")
    
    # Load ground truth games
    num_games = debate_config.get_num_games()
    print(f"\n📚 Loading games {game_id_range[0]}-{game_id_range[1]} of size {game_size} ({num_games} games)...")
    games = load_ground_truth_games(game_size, game_id_range)
    
    if not games:
        print("❌ No games loaded. Please generate ground truth data first.")
        return
    
    print(f"✅ Loaded {len(games)} games")
    
    # Initialize debate system
    print(f"\n🔧 Initializing debate system...")
    debate_system = MultiAgentDebateSystem(debate_config)
    
    # Run debates
    print(f"\n🎯 Running debates...")
    sessions = run_debates_with_system(debate_system, games, use_parallel)
    
    print(f"\n✅ Flexible debate system completed successfully!")
    # Show the base agent folder path
    agent_count = len(debate_config.agents)
    model_names = [agent.model for agent in debate_config.agents]
    conf_suffix = "true" if debate_config.self_reported_confidence else "false"
    agent_folder = f"agent{agent_count}_{'_'.join(model_names)}_conf_{conf_suffix}"
    base_path = os.path.join(debate_config.output_path, agent_folder)
    print(f"📁 Results saved to: {base_path}")

def parse_key_value_args(args):
    """Parse key=value arguments from command line."""
    parsed = {}
    for arg in args:
        if '=' in arg:
            key, value = arg.split('=', 1)
            parsed[key] = value
    return parsed

def main():
    """Main function with command line interface."""
    
    if len(sys.argv) < 2:
        print("Usage: python run_debate.py <command> [options]")
        print("\nCommands:")
        print("  run [game_size] [num_games]")
        print("  custom <agent_configs_json> [game_size] [num_games]")
        print("  flexible <llm_configs_json> [game_size] [num_games]")
        print("\nNew format (recommended):")
        print("  python run_debate.py run game_size=<size> game_id_range=<start,end> [use_parallel=<true/false>] [game_parallel_workers=<num>] [self_reported_confidence=<true/false>]")
        print("  python run_debate.py flexible llm_configs='<json>' game_size=<size> game_id_range=<start,end> [use_parallel=<true/false>] [self_reported_confidence=<true/false>]")
        print("\nExamples:")
        print("  python run_debate.py run game_size=5 game_id_range=1,20")
        print("  python run_debate.py run game_size=5 game_id_range=1,20 use_parallel=false")
        print("  python run_debate.py run game_size=5 game_id_range=1,20 game_parallel_workers=10")
        print("  python run_debate.py run game_size=5 game_id_range=1,20 self_reported_confidence=true")
        print("  python run_debate.py flexible llm_configs='[{\"provider\":\"openai\",\"model\":\"gpt-5-nano\"}]' game_size=5 game_id_range=1,20")
        print("  python run_debate.py run 5 1,20")
        print("  python run_debate.py custom '[{\"name\":\"GPT-5\",\"provider\":\"openai\",\"model\":\"gpt-5-nano\"}]' game_size=5 game_num=1")
        print("  python run_debate.py flexible '[{\"provider\":\"openai\",\"model\":\"gpt-5-nano\"},{\"provider\":\"gemini\",\"model\":\"gemini-2.5-flash\",\"temperature\":0.2}]' game_size=5 game_num=1")
        print("  # Note: temperature is optional for all models")
        print("  # Note: self_reported_confidence enables confidence scoring (0-100) for all model outputs")
        return
    
    # Check if using new key=value format
    kv_args = parse_key_value_args(sys.argv[1:])
    
    if 'game_size' in kv_args or 'game_id_range' in kv_args:
        # New format: python run_debate.py run game_size=X game_id_range=Y,Z
        command = sys.argv[1] if sys.argv[1] not in kv_args else "run"
        game_size = int(kv_args.get('game_size', 5))
        
        # Parse game_id_range
        if 'game_id_range' in kv_args:
            game_id_range_str = kv_args.get('game_id_range', '1,1')
            try:
                start_id, end_id = map(int, game_id_range_str.split(','))
                game_id_range = [start_id, end_id]
            except ValueError:
                print(f"Error: Invalid game_id_range format '{game_id_range_str}'. Use 'start,end' format.")
                return
        else:
            game_id_range = [1, 1]  # Default to single game
        
        if command == "run":
            use_parallel = kv_args.get('use_parallel', 'true').lower() == 'true'
            game_parallel_workers = int(kv_args.get('game_parallel_workers', 20))
            self_reported_confidence = kv_args.get('self_reported_confidence', 'false').lower() == 'true'
            num_games = game_id_range[1] - game_id_range[0] + 1
            print(f"🎯 Running default configuration with {game_size} players, games {game_id_range[0]}-{game_id_range[1]} ({num_games} games)")
            print(f"🔄 Parallel processing: {'enabled' if use_parallel else 'disabled'}")
            print(f"🔧 Game parallel workers: {game_parallel_workers}")
            print(f"📊 Self-reported confidence: {'enabled' if self_reported_confidence else 'disabled'}")
            run_single_debate_session(game_size, game_id_range, use_parallel=use_parallel, game_parallel_workers=game_parallel_workers, self_reported_confidence=self_reported_confidence)
        elif command == "flexible":
            # Support flexible command with key=value format
            if 'llm_configs' not in kv_args:
                print("Error: llm_configs is required for flexible command")
                return
            
            try:
                llm_configs = json.loads(kv_args['llm_configs'])
            except json.JSONDecodeError as e:
                print(f"Error: Invalid JSON for llm_configs: {e}")
                return
            
            use_parallel = kv_args.get('use_parallel', 'true').lower() == 'true'
            self_reported_confidence = kv_args.get('self_reported_confidence', 'false').lower() == 'true'
            num_games = game_id_range[1] - game_id_range[0] + 1
            
            print(f"🎯 Running flexible configuration with {game_size} players, games {game_id_range[0]}-{game_id_range[1]} ({num_games} games)")
            print(f"🤖 Agents: {[config.get('provider', 'unknown') + '-' + config.get('model', 'unknown') for config in llm_configs]}")
            print(f"🔄 Parallel processing: {'enabled' if use_parallel else 'disabled'}")
            print(f"📊 Self-reported confidence: {'enabled' if self_reported_confidence else 'disabled'}")
            
            # Create debate config with custom LLM configs
            debate_config = create_flexible_debate_config(llm_configs, game_size, game_id_range)
            debate_config.self_reported_confidence = self_reported_confidence
            
            # Load ground truth games
            games = load_ground_truth_games(game_size, game_id_range)
            if not games:
                print("❌ No games loaded. Please generate ground truth data first.")
                return
            
            print(f"✅ Loaded {len(games)} games")
            
            # Initialize debate system
            print(f"\n🔧 Initializing debate system...")
            debate_system = MultiAgentDebateSystem(debate_config)
            
            # Run debates
            print(f"\n🎯 Running debates...")
            sessions = run_debates_with_system(debate_system, games, use_parallel)
            
            
            print(f"\n🎉 Debate completed successfully!")
            print(f"📁 Results saved to: {debate_config.get_organized_output_path()}")
        else:
            print(f"Error: Command '{command}' not supported with key=value format")
        return
    
    command = sys.argv[1]
    
    if command == "run":
        game_size = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        # Support both old format (num_games) and new format (game_id_range)
        if len(sys.argv) > 3:
            arg3 = sys.argv[3]
            if ',' in arg3:
                # New format: game_id_range
                try:
                    start_id, end_id = map(int, arg3.split(','))
                    game_id_range = [start_id, end_id]
                except ValueError:
                    print(f"Error: Invalid game_id_range format '{arg3}'. Use 'start,end' format.")
                    return
            else:
                # Old format: num_games (convert to game_id_range)
                num_games = int(arg3)
                game_id_range = [1, num_games]
        else:
            game_id_range = [1, 3]  # Default
        
        run_single_debate_session(game_size, game_id_range)
        
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
        
    elif command == "flexible":
        if len(sys.argv) < 3:
            print("Error: Please provide LLM configurations as JSON string")
            return
        
        try:
            llm_configs = json.loads(sys.argv[2])
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON for LLM configurations: {e}")
            return
        
        game_size = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        num_games = int(sys.argv[4]) if len(sys.argv) > 4 else 3
        
        run_flexible_debate(llm_configs, game_size, num_games)
    
    else:
        print(f"Error: Unknown command '{command}'")
        print("Use 'python run_debate.py' without arguments to see usage information.")

if __name__ == "__main__":
    main()
