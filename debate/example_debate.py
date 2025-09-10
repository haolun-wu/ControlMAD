#!/usr/bin/env python3
"""
Example script demonstrating the multi-agent debate system.
"""

import os
import sys
import json
from typing import List, Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_types import ground_truth
from debate.debate_config import create_custom_debate_config
from debate.debate_system import MultiAgentDebateSystem
from debate.debate_visualizer import DebateVisualizer

def create_sample_game() -> ground_truth:
    """Create a sample game for demonstration."""
    
    # Sample game text
    game_text = """---
Player name: Alice
Player statement: I am a knight.
---
Player name: Bob
Player statement: Alice is lying.
---
Player name: Charlie
Player statement: Among Alice, Bob, and I, there is exactly one spy.
---
Message from the game manager: I am the game manager and here is a hint for you: Among all players, there is exactly one spy.
"""
    
    # Sample solution
    text_solution = """Alice is a knave.
Bob is a knight.
Charlie is a spy.
"""
    
    return ground_truth(
        game_id=999,
        num_player=3,
        num_spy=1,
        num_hint=1,
        raw_schema=[1, 4, 11],
        raw_statement=[[0], [0, 1], [0, 1, 2], [0, 1, 2]],
        raw_solution=[2, 1, 3],  # Alice=knave, Bob=knight, Charlie=spy
        text_game=game_text,
        text_solution=text_solution
    )

def run_example_debate():
    """Run an example debate with a sample game."""
    
    print("🎯 Multi-Agent Debate System Example")
    print("=" * 50)
    
    # Create sample game
    print("📝 Creating sample game...")
    sample_game = create_sample_game()
    
    print("Game Text:")
    print(sample_game.text_game)
    print("\nGround Truth Solution:")
    print(sample_game.text_solution)
    
    # Create debate configuration with 3 different agents
    print("\n🤖 Setting up 3-agent debate...")
    agent_configs = [
        {
            "name": "GPT-5-Analyst",
            "provider": "openai",
            "model": "gpt-5-nano",
            "temperature": 0.7,
            "reasoning_effort": "high",
            "verbosity": "high"
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
    
    debate_config = create_custom_debate_config(agent_configs)
    print(f"Agents: {[agent.name for agent in debate_config.agents]}")
    
    # Initialize debate system
    print("\n🔧 Initializing debate system...")
    debate_system = MultiAgentDebateSystem(debate_config)
    
    # Run debate session
    print("\n🎯 Running debate session...")
    print("=" * 30)
    
    session = debate_system.run_debate_session(sample_game)
    
    # Print detailed results
    print("\n📊 DEBATE RESULTS")
    print("=" * 30)
    
    print(f"\n🎯 Game {session.game_id} - Final Results:")
    print(f"Ground Truth: {session.ground_truth_solution}")
    
    if session.final_vote:
        print(f"Majority Vote: {session.final_vote}")
        
        # Check accuracy
        correct = 0
        total = len(session.ground_truth_solution)
        for player, role in session.ground_truth_solution.items():
            if session.final_vote.get(player) == role:
                correct += 1
        accuracy = correct / total if total > 0 else 0
        print(f"Accuracy: {accuracy:.2%}")
    
    if session.supervisor_decision:
        print(f"Supervisor Decision: {session.supervisor_decision}")
    
    # Print round-by-round details
    print(f"\n📋 Round-by-Round Analysis:")
    for round_data in session.debate_rounds:
        print(f"\n--- Round {round_data.round_number}: {round_data.player_name} ---")
        print(f"Consensus Reached: {round_data.consensus_reached}")
        if round_data.majority_role:
            print(f"Majority Role: {round_data.majority_role}")
        
        print("Agent Responses:")
        for response in round_data.agent_responses:
            role = response.player_role_assignments.get(round_data.player_name, "unknown")
            print(f"  {response.agent_name}: {role}")
            print(f"    Reasoning: {response.explanation[:100]}...")
    
    # Create visualizations
    print(f"\n🎨 Creating visualizations...")
    visualizer = DebateVisualizer(debate_config.output_path)
    visualization_paths = visualizer.create_all_visualizations([session])
    
    print(f"\n📊 Visualization files created:")
    for name, path in visualization_paths.items():
        print(f"  - {name}: {path}")
    
    # Performance tracking
    if session.performance_tracking:
        print(f"\n📈 Performance Tracking:")
        tracking = session.performance_tracking
        
        print("Initial Accuracy:")
        for agent, acc in tracking.get('initial_accuracy', {}).items():
            print(f"  {agent}: {acc:.2%}")
        
        print("Final Accuracy:")
        for method, acc in tracking.get('final_accuracy', {}).items():
            print(f"  {method}: {acc:.2%}")
    
    print(f"\n✅ Example debate completed successfully!")
    print(f"📁 All results saved to: {debate_config.output_path}")

def run_comparison_experiment():
    """Run a comparison between single-agent and multi-agent approaches."""
    
    print("\n🔬 Comparison Experiment: Single vs Multi-Agent")
    print("=" * 50)
    
    # Create sample games
    sample_games = [create_sample_game()]
    
    # Single-agent baseline (using existing test system)
    print("\n1️⃣ Running single-agent baseline...")
    from test_baseline import Test
    from config import test_config
    
    # Configure for single agent
    test_config.provider = "openai"
    test_config.model = "gpt-5-nano"
    test_config.num_worker = 1
    
    single_agent_test = Test(test_config)
    single_agent_test.groundtruth = sample_games
    
    # Run single agent test
    single_results = single_agent_test.run_parallel_test_with_config()
    single_accuracy = single_agent_test.verify_test_results()
    
    print(f"Single-agent accuracy: {single_accuracy}/{len(sample_games)}")
    
    # Multi-agent debate
    print("\n2️⃣ Running multi-agent debate...")
    agent_configs = [
        {"name": "Agent-1", "provider": "openai", "model": "gpt-5-nano"},
        {"name": "Agent-2", "provider": "gemini", "model": "gemini-2.5-flash-lite"},
        {"name": "Agent-3", "provider": "ali", "model": "qwen-flash"}
    ]
    
    debate_config = create_custom_debate_config(agent_configs)
    debate_system = MultiAgentDebateSystem(debate_config)
    
    debate_sessions = debate_system.run_batch_debate(sample_games)
    
    # Calculate multi-agent accuracy
    multi_accuracy = 0
    for session in debate_sessions:
        if session.final_vote:
            correct = 0
            total = len(session.ground_truth_solution)
            for player, role in session.ground_truth_solution.items():
                if session.final_vote.get(player) == role:
                    correct += 1
            if correct == total:
                multi_accuracy += 1
    
    print(f"Multi-agent accuracy: {multi_accuracy}/{len(sample_games)}")
    
    # Comparison summary
    print(f"\n📊 COMPARISON SUMMARY:")
    print(f"Single-agent: {single_accuracy}/{len(sample_games)} ({single_accuracy/len(sample_games)*100:.1f}%)")
    print(f"Multi-agent:  {multi_accuracy}/{len(sample_games)} ({multi_accuracy/len(sample_games)*100:.1f}%)")
    
    if multi_accuracy > single_accuracy:
        print("🎉 Multi-agent approach performed better!")
    elif multi_accuracy == single_accuracy:
        print("🤝 Both approaches performed equally well.")
    else:
        print("🤔 Single-agent approach performed better.")

def main():
    """Main function."""
    
    if len(sys.argv) > 1 and sys.argv[1] == "comparison":
        run_comparison_experiment()
    else:
        run_example_debate()

if __name__ == "__main__":
    main()
