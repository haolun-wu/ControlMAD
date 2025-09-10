#!/usr/bin/env python3
"""
Test script for the multi-agent debate system.
"""

import os
import sys
import json
from typing import List, Dict, Any

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from project_types import ground_truth
from debate_config import create_custom_debate_config
from debate_system import MultiAgentDebateSystem

def test_debate_config():
    """Test debate configuration creation."""
    print("🧪 Testing debate configuration...")
    
    # Test custom configuration
    agent_configs = [
        {"name": "Test-Agent-1", "provider": "openai", "model": "gpt-5-nano"},
        {"name": "Test-Agent-2", "provider": "gemini", "model": "gemini-2.5-flash-lite"}
    ]
    
    config = create_custom_debate_config(agent_configs)
    
    assert len(config.agents) == 2
    assert config.agents[0].name == "Test-Agent-1"
    assert config.agents[1].name == "Test-Agent-2"
    assert config.enable_self_adjustment == True
    assert config.enable_majority_vote == True
    
    print("✅ Debate configuration test passed")

def test_agent_response_parsing():
    """Test agent response parsing."""
    print("🧪 Testing agent response parsing...")
    
    from debate_system import MultiAgentDebateSystem
    
    # Create a minimal config for testing
    config = create_custom_debate_config([
        {"name": "Test-Agent", "provider": "openai", "model": "gpt-5-nano"}
    ])
    
    # Initialize system (this will fail without API keys, but we can test parsing)
    try:
        system = MultiAgentDebateSystem(config)
    except Exception:
        # Expected to fail without API keys, but we can test parsing methods
        system = MultiAgentDebateSystem.__new__(MultiAgentDebateSystem)
    
    # Test JSON response parsing
    json_response = '''
    {
        "players": [
            {"name": "Alice", "role": "knight"},
            {"name": "Bob", "role": "knave"},
            {"name": "Charlie", "role": "spy"}
        ],
        "explanation": "Alice is a knight because she tells the truth."
    }
    '''
    
    player_assignments, explanation = system._parse_agent_response(json_response)
    
    assert player_assignments["Alice"] == "knight"
    assert player_assignments["Bob"] == "knave" 
    assert player_assignments["Charlie"] == "spy"
    assert "Alice is a knight" in explanation
    
    print("✅ Agent response parsing test passed")

def test_ground_truth_parsing():
    """Test ground truth solution parsing."""
    print("🧪 Testing ground truth parsing...")
    
    from debate_system import MultiAgentDebateSystem
    
    # Create a minimal config
    config = create_custom_debate_config([
        {"name": "Test-Agent", "provider": "openai", "model": "gpt-5-nano"}
    ])
    
    try:
        system = MultiAgentDebateSystem(config)
    except Exception:
        system = MultiAgentDebateSystem.__new__(MultiAgentDebateSystem)
    
    # Test ground truth parsing
    sample_gt = ground_truth(
        game_id=1,
        num_player=3,
        num_spy=1,
        num_hint=1,
        raw_schema=[1, 2, 3],
        raw_statement=[[0], [1], [2]],
        raw_solution=[1, 2, 3],
        text_game="Sample game text",
        text_solution="Alice is a knight.\nBob is a knave.\nCharlie is a spy.\n"
    )
    
    gt_solution = system._parse_ground_truth_solution(sample_gt)
    
    assert gt_solution["Alice"] == "knight"
    assert gt_solution["Bob"] == "knave"
    assert gt_solution["Charlie"] == "spy"
    
    print("✅ Ground truth parsing test passed")

def test_consensus_checking():
    """Test consensus checking logic."""
    print("🧪 Testing consensus checking...")
    
    from debate_system import MultiAgentDebateSystem, AgentResponse
    from datetime import datetime
    
    # Create a minimal config
    config = create_custom_debate_config([
        {"name": "Test-Agent", "provider": "openai", "model": "gpt-5-nano"}
    ])
    
    try:
        system = MultiAgentDebateSystem(config)
    except Exception:
        system = MultiAgentDebateSystem.__new__(MultiAgentDebateSystem)
    
    # Test consensus with majority
    responses = [
        AgentResponse(
            agent_name="Agent-1",
            game_id=1,
            round_number=1,
            phase="debate",
            player_role_assignments={"Alice": "knight"},
            explanation="Test",
            timestamp=datetime.now().isoformat()
        ),
        AgentResponse(
            agent_name="Agent-2", 
            game_id=1,
            round_number=1,
            phase="debate",
            player_role_assignments={"Alice": "knight"},
            explanation="Test",
            timestamp=datetime.now().isoformat()
        ),
        AgentResponse(
            agent_name="Agent-3",
            game_id=1,
            round_number=1,
            phase="debate", 
            player_role_assignments={"Alice": "knight"},
            explanation="Test",
            timestamp=datetime.now().isoformat()
        )
    ]
    
    consensus = system._check_consensus(responses, "Alice")
    assert consensus == "knight"
    
    # Test no consensus
    responses[2].player_role_assignments["Alice"] = "knave"
    consensus = system._check_consensus(responses, "Alice")
    assert consensus is None
    
    print("✅ Consensus checking test passed")

def test_performance_tracking():
    """Test performance tracking creation."""
    print("🧪 Testing performance tracking...")
    
    from debate_system import MultiAgentDebateSystem, AgentResponse, DebateRound, DebateSession
    from datetime import datetime
    
    # Create a minimal config
    config = create_custom_debate_config([
        {"name": "Test-Agent", "provider": "openai", "model": "gpt-5-nano"}
    ])
    
    try:
        system = MultiAgentDebateSystem(config)
    except Exception:
        system = MultiAgentDebateSystem.__new__(MultiAgentDebateSystem)
    
    # Create test data
    sample_gt = ground_truth(
        game_id=1,
        num_player=2,
        num_spy=1,
        num_hint=1,
        raw_schema=[1, 2],
        raw_statement=[[0], [1]],
        raw_solution=[1, 3],
        text_game="Sample game",
        text_solution="Alice is a knight.\nBob is a spy.\n"
    )
    
    initial_proposals = [
        AgentResponse(
            agent_name="Agent-1",
            game_id=1,
            round_number=0,
            phase="initial",
            player_role_assignments={"Alice": "knight", "Bob": "spy"},
            explanation="Test",
            timestamp=datetime.now().isoformat()
        )
    ]
    
    debate_rounds = [
        DebateRound(
            player_name="Alice",
            round_number=1,
            agent_responses=[
                AgentResponse(
                    agent_name="Agent-1",
                    game_id=1,
                    round_number=1,
                    phase="debate",
                    player_role_assignments={"Alice": "knight", "Bob": "spy"},
                    explanation="Test",
                    timestamp=datetime.now().isoformat()
                )
            ]
        )
    ]
    
    final_vote = {"Alice": "knight", "Bob": "spy"}
    
    # Test performance tracking
    tracking = system._create_performance_tracking(
        sample_gt, initial_proposals, debate_rounds, final_vote, None
    )
    
    assert tracking["game_id"] == 1
    assert "initial_accuracy" in tracking
    assert "final_accuracy" in tracking
    assert "round_by_round" in tracking
    assert tracking["ground_truth"] == {"Alice": "knight", "Bob": "spy"}
    
    print("✅ Performance tracking test passed")

def run_all_tests():
    """Run all tests."""
    print("🚀 Running Multi-Agent Debate System Tests")
    print("=" * 50)
    
    try:
        test_debate_config()
        test_agent_response_parsing()
        test_ground_truth_parsing()
        test_consensus_checking()
        test_performance_tracking()
        
        print("\n🎉 All tests passed successfully!")
        print("The multi-agent debate system is ready to use.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
