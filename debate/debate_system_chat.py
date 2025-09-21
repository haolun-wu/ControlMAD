"""
Enhanced Multi-Agent Debate System with Chat History

This module extends the original debate system to use chat histories for better
agent self-awareness and natural multi-turn dialogue format.
"""

import json
import os
import time
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.project_types import ground_truth, response_format, token_usage
from debate.debate_config import DebateConfig, AgentConfig
from debate.debate_system import MultiAgentDebateSystem, AgentResponse, DebateRound, DebateSession
from debate.agent_chat_manager import AgentChatManager, MessageRole
from utils.utility import openai_client, gemini_client, ali_client, cstcloud, ParallelProcessor
from prompts import (
    kks_system_prompt,
    kks_response_schema,
    get_kks_system_prompt_with_confidence,
    get_kks_response_schema_with_confidence,
    get_kks_debate_response_schema_with_confidence
)


class ChatHistoryDebateSystem(MultiAgentDebateSystem):
    """
    Enhanced Multi-Agent Debate System with Chat History Support.
    
    This system maintains individual chat histories for each agent, enabling
    better self-awareness through clear separation of agent messages and
    natural multi-turn dialogue format.
    """
    
    def __init__(self, debate_config: DebateConfig, secret_path: str = "secret.json", setup_logging: bool = True):
        super().__init__(debate_config, secret_path, setup_logging)
        self.chat_manager: Optional[AgentChatManager] = None
    
    def run_debate_session(self, game: ground_truth) -> DebateSession:
        """Run a complete debate for a single game with chat history support."""
        # Initialize chat manager for this game
        agent_names = [name for name in self.agents.keys()]
        self.chat_manager = AgentChatManager(game.game_id, agent_names)
        
        # Print current log file to console for user awareness
        print(f"\n🔄 Now running Game {game.game_id} with Chat History")
        print(f"📁 Current log: {self.organized_output_path}/debate_log_game{game.game_id}_*.log")
        print("=" * 60)
        
        self.logger.info(f"🎯 Starting chat history debate for Game {game.game_id}")
        self.logger.info(f"Players: {[f'Player {i+1}' for i in range(game.num_player)]}")
        
        # Parse ground truth solution
        gt_solution = self._parse_ground_truth_solution(game)
        self.logger.info(f"📋 Ground Truth Solution: {gt_solution}")
        
        # Phase 1: Initial proposals with chat history
        self.logger.info("📝 Phase 1: Initial Proposals (Chat History)")
        self.logger.info("=" * 40)
        initial_proposals = self._get_initial_proposals_with_chat(game)
        
        # Show initial proposals
        self.logger.info("🔍 Initial Proposals Summary:")
        for proposal in initial_proposals:
            self.logger.info(f"  {proposal.agent_name}: {proposal.player_role_assignments}")
            if proposal.error:
                self.logger.error(f"    ❌ Error: {proposal.error}")
            else:
                self.logger.info(f"    ✅ Success")
            
            # Log per-player accuracy for initial proposals
            self.logger.info(f"    📊 {proposal.agent_name} Initial Accuracy by Player:")
            for player, gt_role in gt_solution.items():
                predicted_role = proposal.player_role_assignments.get(player, "unknown")
                correct = "✅" if predicted_role == gt_role else "❌"
                self.logger.info(f"      {player}: {predicted_role} vs {gt_role} {correct}")
        
        # Phase 2: Debate rounds for each player with chat history
        self.logger.info("🗣️ Phase 2: Debate Rounds (Chat History)")
        debate_rounds = []
        
        # Get player names from ground truth
        player_names = list(gt_solution.keys())
        
        # Track the current state of all agents (starts with initial proposals)
        current_agent_states = initial_proposals
        
        for player_name in player_names:
            self.logger.info(f"--- Debating {player_name}'s role ---")
            self.logger.info(f"🎯 Ground Truth: {player_name} is a {gt_solution.get(player_name, 'unknown')}")
            debate_round = self._run_debate_round_with_chat(
                game, player_name, len(debate_rounds) + 1, 
                current_agent_states, debate_rounds
            )
            debate_rounds.append(debate_round)
            
            # Update current agent states to the self-adjustment results from this round
            current_agent_states = debate_round.agent_responses
            
            # Show round results
            self.logger.info(f"📊 Round {debate_round.round_number} Results for {player_name}:")
            for response in debate_round.agent_responses:
                role = response.player_role_assignments.get(player_name, "unknown")
                correct = "✅" if role == gt_solution.get(player_name) else "❌"
                confidence_info = f" (confidence: {response.confidence})" if self.config.self_reported_confidence and response.confidence > 0 else ""
                self.logger.info(f"  {response.agent_name}: {role} {correct}{confidence_info}")
                
                # Log detailed per-player accuracy for this round
                self.logger.info(f"    📊 {response.agent_name} Round {debate_round.round_number} Accuracy by Player:")
                for player, gt_role in gt_solution.items():
                    predicted_role = response.player_role_assignments.get(player, "unknown")
                    correct = "✅" if predicted_role == gt_role else "❌"
                    self.logger.info(f"      {player}: {predicted_role} vs {gt_role} {correct}")
            self.logger.info(f"🎯 Round completed - no intermediate voting")
        
        # Phase 3: Final Discussion and Fresh Voting with chat history
        self.logger.info("🗣️ Phase 3: Final Discussion and Fresh Voting (Chat History)")
        self.logger.info("=" * 40)
        final_vote = self._conduct_final_discussion_and_vote_with_chat(game, initial_proposals, debate_rounds)
        
        # Show final vote results
        self.logger.info(f"📊 Final Vote Results:")
        for player, role in final_vote.items():
            correct = "✅" if role == gt_solution.get(player) else "❌"
            self.logger.info(f"  {player}: {role} {correct}")
        
        # Phase 4: Supervisor decision if needed (unchanged)
        supervisor_decision = None
        if not self._is_consensus_reached(final_vote):
            self.logger.info("👨‍💼 Phase 4: Supervisor Decision")
            self.logger.info("=" * 40)
            supervisor_decision = self._get_supervisor_decision(game, initial_proposals, debate_rounds)
            
            if supervisor_decision:
                self.logger.info(f"📊 Supervisor Decision:")
                for player, role in supervisor_decision.items():
                    correct = "✅" if role == gt_solution.get(player) else "❌"
                    self.logger.info(f"  {player}: {role} {correct}")
        
        # Create performance tracking
        performance_tracking = self._create_performance_tracking(
            game, initial_proposals, debate_rounds, final_vote, supervisor_decision
        )
        
        # Create debate session
        session = DebateSession(
            game_id=game.game_id,
            game_text=game.text_game,
            ground_truth_solution=gt_solution,
            initial_proposals=initial_proposals,
            debate_rounds=debate_rounds,
            final_vote=final_vote,
            supervisor_decision=supervisor_decision,
            performance_tracking=performance_tracking
        )
        
        # Save debate session
        self._save_debate_session(session)
        
        # Save chat histories
        self._save_chat_histories(session)
        
        # Final accuracy summary
        self.logger.info(f"📈 FINAL ACCURACY SUMMARY")
        self.logger.info("=" * 40)
        
        # Calculate final accuracy
        final_solution = supervisor_decision if supervisor_decision else final_vote
        if final_solution:
            correct = 0
            total = len(gt_solution)
            for player, role in gt_solution.items():
                if final_solution.get(player) == role:
                    correct += 1
            accuracy = correct / total if total > 0 else 0
            self.logger.info(f"🎯 Final Accuracy: {accuracy:.2%} ({correct}/{total})")
            
            # Show which players were correct/incorrect
            self.logger.info(f"📋 Player-by-Player Results:")
            for player, gt_role in gt_solution.items():
                final_role = final_solution.get(player, "unknown")
                status = "✅ CORRECT" if final_role == gt_role else "❌ INCORRECT"
                self.logger.info(f"  {player}: {final_role} (GT: {gt_role}) {status}")
        
        self.logger.info(f"✅ Chat history debate completed for Game {game.game_id}")
        
        # Generate visualizations automatically
        self._generate_visualizations(session)
        
        return session
    
    def _get_initial_proposals_with_chat(self, game: ground_truth) -> List[AgentResponse]:
        """Get initial proposals from all agents using chat history."""
        def get_single_proposal_with_chat(agent_name: str) -> AgentResponse:
            agent_info = self.agents[agent_name]
            client = agent_info["client"]
            config = agent_info["config"]
            
            self.logger.info(f"    🤖 {agent_name} ({config.provider}/{config.model}) getting initial proposal with chat history...")
            
            try:
                # Set up initial chat history for this agent
                self.chat_manager.create_initial_system_message(
                    agent_name, game.text_game, game.num_player, self.config.self_reported_confidence
                )
                self.chat_manager.create_game_context_message(agent_name, game.text_game)
                
                # Get chat context for API call
                chat_context = self.chat_manager.get_chat_context_for_agent(agent_name)
                
                # Log initial proposal context
                self.logger.info(f"    📝 {agent_name} CHAT CONTEXT:")
                for i, msg in enumerate(chat_context):
                    self.logger.info(f"      Message {i+1} ({msg['role']}): {msg['content'][:100]}...")
                
                # Make API call based on provider
                if config.provider == "openai":
                    # Check if model supports reasoning parameters
                    if config.model in ["gpt-5-nano", "gpt-5", "gpt-4o"]:
                        response_obj = client.response_completion(
                            user_prompt="",  # Empty user prompt since context is in chat history
                            system_prompt="",  # Empty system prompt since it's in chat history
                            model=config.model,
                            reasoning_effort=config.reasoning_effort,
                            verbosity=config.verbosity,
                            messages=chat_context  # Pass chat context
                        )
                    else:
                        # For models that don't support reasoning, we need to construct a single prompt
                        combined_prompt = self._combine_chat_context_to_prompt(chat_context)
                        response_obj = client.response_completion(
                            user_prompt=combined_prompt,
                            system_prompt="",
                            model=config.model
                        )
                elif config.provider == "gemini":
                    # Gemini supports chat format
                    response_obj = client.chat_completion(
                        user_prompt="",  # Empty user prompt since context is in chat history
                        system_prompt="",  # Empty system prompt since it's in chat history
                        model=config.model,
                        response_schema=get_kks_response_schema_with_confidence(self.config.self_reported_confidence),
                        messages=chat_context  # Pass chat context
                    )
                else:  # ali, cst - these may need single prompt format
                    combined_prompt = self._combine_chat_context_to_prompt(chat_context)
                    response_obj = client.chat_completion(
                        user_prompt=combined_prompt,
                        system_prompt="",
                        model=config.model
                    )
                
                # Parse response
                player_assignments, explanation, confidence, _, _, _, _ = self._parse_agent_response(response_obj.text)
                
                # Add agent's response to their chat history
                response_content = json.dumps({
                    "players": [{"name": name, "role": role} for name, role in player_assignments.items()],
                    "confidence": confidence,
                    "explanation": explanation
                }, indent=2)
                
                self.chat_manager.add_agent_response(
                    agent_name, response_content, "initial"
                )
                
                self.logger.info(f"    ✅ {agent_name} completed: {player_assignments}")
                if self.config.self_reported_confidence:
                    self.logger.info(f"    📊 {agent_name} confidence: {confidence}")
                else:
                    self.logger.info(f"    📊 {agent_name} confidence: not requested (SRC disabled)")
                
                return AgentResponse(
                    agent_name=agent_name,
                    game_id=game.game_id,
                    round_number=0,
                    phase="initial",
                    player_role_assignments=player_assignments,
                    explanation=explanation,
                    confidence=confidence,
                    response_obj=response_obj,
                    timestamp=datetime.now().isoformat(),
                    error=""
                )
                
            except Exception as e:
                self.logger.error(f"    ❌ {agent_name} failed: {e}")
                return AgentResponse(
                    agent_name=agent_name,
                    game_id=game.game_id,
                    round_number=0,
                    phase="initial",
                    player_role_assignments={},
                    explanation=f"Error: {str(e)}",
                    confidence=1.0,  # Low confidence for errors
                    timestamp=datetime.now().isoformat(),
                    error=str(e)
                )
        
        # Get proposals from all agents in parallel
        agent_names = [name for name in self.agents.keys()]
        proposals = self.parallel_processor.process_tasks(
            get_single_proposal_with_chat, agent_names, preserve_order=True
        )
        
        return [p for p in proposals if p is not None]
    
    def _run_debate_round_with_chat(self, game: ground_truth, player_name: str, 
                                   round_num: int, current_agent_states: List[AgentResponse],
                                   previous_rounds: List[DebateRound]) -> DebateRound:
        """Run a debate round for a specific player's role using chat history."""
        
        # Step 1: Debate period with chat history
        self.logger.info(f"  Step 1: Debate period for {player_name} (Chat History)")
        debate_responses = self._conduct_debate_for_player_with_chat(
            game, player_name, round_num, current_agent_states, previous_rounds
        )
        
        # Step 2: Self-adjustment with chat history
        self.logger.info(f"  Step 2: Self-adjustment for {player_name} (Chat History)")
        if self.config.enable_self_adjustment:
            adjusted_responses = self._conduct_self_adjustment_with_chat(
                game, player_name, round_num, debate_responses, previous_rounds, current_agent_states
            )
        else:
            adjusted_responses = debate_responses
        
        # Combine both phases into a single list
        all_responses = debate_responses + adjusted_responses
        
        # Create debate summary
        debate_summary = self._create_debate_summary(adjusted_responses, player_name)
        
        # Check for consensus in self-adjustment responses
        consensus_role = self._check_consensus(adjusted_responses, player_name)
        consensus_reached = consensus_role is not None
        
        if consensus_reached:
            self.logger.info(f"🎯 CONSENSUS REACHED for {player_name}: {consensus_role}")
        else:
            self.logger.info(f"❌ NO CONSENSUS for {player_name} - agents disagree")
        
        return DebateRound(
            player_name=player_name,
            round_number=round_num,
            agent_responses=all_responses,  # Now includes both debate and self_adjustment phases
            debate_summary=debate_summary,
            consensus_reached=consensus_reached,
            majority_role=consensus_role
        )
    
    def _conduct_debate_for_player_with_chat(self, game: ground_truth, player_name: str,
                                           round_num: int, current_agent_states: List[AgentResponse],
                                           previous_rounds: List[DebateRound]) -> List[AgentResponse]:
        """Conduct debate for a specific player's role using chat history."""
        
        def get_debate_response_with_chat(agent_name: str) -> AgentResponse:
            agent_info = self.agents[agent_name]
            client = agent_info["client"]
            config = agent_info["config"]
            
            self.logger.info(f"      🤖 {agent_name} debating {player_name} with chat history...")
            
            try:
                # Create debate context for this agent's chat history
                debate_prompt = self._create_debate_prompt_for_chat(
                    game, player_name, current_agent_states, previous_rounds, agent_name
                )
                
                self.chat_manager.create_debate_context_message(
                    agent_name, debate_prompt, player_name, round_num
                )
                
                # Add other agents' context to this agent's chat history
                other_agents_data = [
                    {
                        "agent_name": state.agent_name,
                        "player_role_assignments": state.player_role_assignments,
                        "explanation": state.explanation,
                        "confidence": state.confidence
                    }
                    for state in current_agent_states
                ]
                
                self.chat_manager.add_other_agents_context(
                    agent_name, other_agents_data, "debate", round_num
                )
                
                # Get chat context for API call
                chat_context = self.chat_manager.get_chat_context_for_agent(agent_name)
                
                # Log debate context
                self.logger.info(f"      📝 {agent_name} DEBATE CHAT CONTEXT for {player_name}:")
                self.logger.info(f"      Total messages: {len(chat_context)}")
                for i, msg in enumerate(chat_context[-3:]):  # Show last 3 messages
                    self.logger.info(f"        Message {len(chat_context)-2+i} ({msg['role']}): {msg['content'][:100]}...")
                
                # Make API call
                if config.provider == "openai":
                    # Check if model supports reasoning parameters
                    if config.model in ["gpt-5-nano", "gpt-5", "gpt-4o"]:
                        response_obj = client.response_completion(
                            user_prompt="",
                            system_prompt="",
                            model=config.model,
                            reasoning_effort=config.reasoning_effort,
                            verbosity=config.verbosity,
                            messages=chat_context
                        )
                    else:
                        # For models that don't support reasoning, combine to single prompt
                        combined_prompt = self._combine_chat_context_to_prompt(chat_context)
                        response_obj = client.response_completion(
                            user_prompt=combined_prompt,
                            system_prompt="",
                            model=config.model
                        )
                elif config.provider == "gemini":
                    # Use debate-specific schema for Gemini
                    debate_response_schema = get_kks_debate_response_schema_with_confidence(self.config.self_reported_confidence)
                    response_obj = client.chat_completion(
                        user_prompt="",
                        system_prompt="",
                        model=config.model,
                        response_schema=debate_response_schema,
                        messages=chat_context
                    )
                else:
                    combined_prompt = self._combine_chat_context_to_prompt(chat_context)
                    response_obj = client.chat_completion(
                        user_prompt=combined_prompt,
                        system_prompt="",
                        model=config.model
                    )
                
                # Parse response
                player_assignments, explanation, confidence, agree_with, disagree_with, agree_reasoning, disagree_reasoning = self._parse_agent_response(response_obj.text, "debate")
                
                # Add agent's debate response to their chat history
                debate_response_content = json.dumps({
                    "player_role": player_name,
                    "role": player_assignments.get(player_name, "unknown"),
                    "agree_with": agree_with or [],
                    "disagree_with": disagree_with or [],
                    "agree_reasoning": agree_reasoning or "",
                    "disagree_reasoning": disagree_reasoning or "",
                    "confidence": confidence
                }, indent=2)
                
                self.chat_manager.add_agent_response(
                    agent_name, debate_response_content, "debate", 
                    player_focus=player_name, round_number=round_num
                )
                
                self.logger.info(f"      ✅ {agent_name} debate completed")
                if self.config.self_reported_confidence:
                    self.logger.info(f"      📊 {agent_name} confidence: {confidence}")
                
                # Log agreement/disagreement analysis
                if agree_with and len(agree_with) > 0:
                    self.logger.info(f"      🤝 {agent_name} agrees with: {', '.join(agree_with)}")
                if disagree_with and len(disagree_with) > 0:
                    self.logger.info(f"      ❌ {agent_name} disagrees with: {', '.join(disagree_with)}")
                
                return AgentResponse(
                    agent_name=agent_name,
                    game_id=game.game_id,
                    round_number=round_num,
                    phase="debate",
                    player_role_assignments=player_assignments,
                    explanation=explanation,
                    confidence=confidence,
                    response_obj=response_obj,
                    timestamp=datetime.now().isoformat(),
                    error="",
                    agree_with=agree_with,
                    disagree_with=disagree_with,
                    agree_reasoning=agree_reasoning,
                    disagree_reasoning=disagree_reasoning
                )
                
            except Exception as e:
                self.logger.error(f"      ❌ {agent_name} debate failed: {e}")
                return AgentResponse(
                    agent_name=agent_name,
                    game_id=game.game_id,
                    round_number=round_num,
                    phase="debate",
                    player_role_assignments={},
                    explanation=f"Error: {str(e)}",
                    confidence=1.0,  # Low confidence for errors
                    timestamp=datetime.now().isoformat(),
                    error=str(e),
                    agree_with=None,
                    disagree_with=None,
                    agree_reasoning=None,
                    disagree_reasoning=None
                )
        
        # Get debate responses from all agents
        agent_names = [name for name in self.agents.keys()]
        responses = self.parallel_processor.process_tasks(
            get_debate_response_with_chat, agent_names, preserve_order=True
        )
        
        return [r for r in responses if r is not None]
    
    def _conduct_self_adjustment_with_chat(self, game: ground_truth, player_name: str,
                                         round_num: int, debate_responses: List[AgentResponse],
                                         previous_rounds: List[DebateRound], 
                                         current_agent_states: List[AgentResponse]) -> List[AgentResponse]:
        """Conduct self-adjustment phase for agents using chat history."""
        
        def get_adjustment_response_with_chat(agent_name: str) -> AgentResponse:
            agent_info = self.agents[agent_name]
            client = agent_info["client"]
            config = agent_info["config"]
            
            self.logger.info(f"      🔄 {agent_name} self-adjusting for {player_name} with chat history...")
            
            try:
                # Create self-adjustment context for this agent's chat history
                adjustment_prompt = self._create_self_adjustment_prompt_for_chat(
                    game, player_name, debate_responses, previous_rounds, agent_name, config, current_agent_states
                )
                
                self.chat_manager.create_self_adjustment_context_message(
                    agent_name, adjustment_prompt, player_name, round_num
                )
                
                # Add debate responses context to this agent's chat history
                debate_responses_data = [
                    {
                        "agent_name": response.agent_name,
                        "player_role_assignments": response.player_role_assignments,
                        "agree_with": response.agree_with,
                        "disagree_with": response.disagree_with,
                        "agree_reasoning": response.agree_reasoning,
                        "disagree_reasoning": response.disagree_reasoning,
                        "confidence": response.confidence
                    }
                    for response in debate_responses
                ]
                
                self.chat_manager.add_other_agents_context(
                    agent_name, debate_responses_data, "self_adjustment", round_num
                )
                
                # Get chat context for API call
                chat_context = self.chat_manager.get_chat_context_for_agent(agent_name)
                
                # Log self-adjustment context
                self.logger.info(f"      📝 {agent_name} SELF-ADJUSTMENT CHAT CONTEXT for {player_name}:")
                self.logger.info(f"      Total messages: {len(chat_context)}")
                
                # Make API call
                if config.provider == "openai":
                    # Check if model supports reasoning parameters
                    if config.model in ["gpt-5-nano", "gpt-5", "gpt-4o"]:
                        response_obj = client.response_completion(
                            user_prompt="",
                            system_prompt="",
                            model=config.model,
                            reasoning_effort=config.reasoning_effort,
                            verbosity=config.verbosity,
                            messages=chat_context
                        )
                    else:
                        # For models that don't support reasoning, combine to single prompt
                        combined_prompt = self._combine_chat_context_to_prompt(chat_context)
                        response_obj = client.response_completion(
                            user_prompt=combined_prompt,
                            system_prompt="",
                            model=config.model
                        )
                elif config.provider == "gemini":
                    response_obj = client.chat_completion(
                        user_prompt="",
                        system_prompt="",
                        model=config.model,
                        response_schema=get_kks_response_schema_with_confidence(self.config.self_reported_confidence),
                        messages=chat_context
                    )
                else:
                    combined_prompt = self._combine_chat_context_to_prompt(chat_context)
                    response_obj = client.chat_completion(
                        user_prompt=combined_prompt,
                        system_prompt="",
                        model=config.model
                    )
                
                # Parse response
                player_assignments, explanation, confidence, _, _, _, _ = self._parse_agent_response(response_obj.text, "self_adjustment")
                
                # Add agent's self-adjustment response to their chat history
                adjustment_response_content = json.dumps({
                    "players": [{"name": name, "role": role} for name, role in player_assignments.items()],
                    "confidence": confidence,
                    "explanation": explanation
                }, indent=2)
                
                self.chat_manager.add_agent_response(
                    agent_name, adjustment_response_content, "self_adjustment",
                    player_focus=player_name, round_number=round_num
                )
                
                self.logger.info(f"      ✅ {agent_name} self-adjustment completed")
                if self.config.self_reported_confidence:
                    self.logger.info(f"      📊 {agent_name} confidence: {confidence}")
                
                return AgentResponse(
                    agent_name=agent_name,
                    game_id=game.game_id,
                    round_number=round_num,
                    phase="self_adjustment",
                    player_role_assignments=player_assignments,
                    explanation=explanation,
                    confidence=confidence,
                    response_obj=response_obj,
                    timestamp=datetime.now().isoformat(),
                    error=""
                )
                
            except Exception as e:
                self.logger.error(f"      ❌ {agent_name} self-adjustment failed: {e}")
                return AgentResponse(
                    agent_name=agent_name,
                    game_id=game.game_id,
                    round_number=round_num,
                    phase="self_adjustment",
                    player_role_assignments={},
                    explanation=f"Error: {str(e)}",
                    confidence=1.0,  # Low confidence for errors
                    timestamp=datetime.now().isoformat(),
                    error=str(e)
                )
        
        # Get adjustment responses from all agents
        agent_names = [name for name in self.agents.keys()]
        responses = self.parallel_processor.process_tasks(
            get_adjustment_response_with_chat, agent_names, preserve_order=True
        )
        
        return [r for r in responses if r is not None]
    
    def _conduct_final_discussion_and_vote_with_chat(self, game: ground_truth, initial_proposals: List[AgentResponse], debate_rounds: List[DebateRound]) -> Dict[str, str]:
        """Conduct final discussion where agents reconsider everything using chat history."""
        self.logger.info("Conducting final discussion and fresh voting with chat history...")
        
        # Get all player names
        player_names = list(initial_proposals[0].player_role_assignments.keys())
        
        # Create final discussion context for each agent's chat history
        final_discussion_prompt = self._create_final_discussion_prompt_for_chat(game, initial_proposals, debate_rounds)
        
        # Get fresh votes from all agents
        fresh_votes = []
        agent_names = [name for name in self.agents.keys()]
        
        for agent_name in agent_names:
            agent_info = self.agents[agent_name]
            client = agent_info["client"]
            config = agent_info["config"]
            
            self.logger.info(f"  🤖 {agent_name} making final decision with chat history...")
            
            try:
                # Add final discussion context to this agent's chat history
                self.chat_manager.create_final_discussion_message(agent_name, final_discussion_prompt)
                
                # Add all previous responses as context
                all_responses_data = []
                
                # Add initial proposals
                for proposal in initial_proposals:
                    all_responses_data.append({
                        "agent_name": proposal.agent_name,
                        "player_role_assignments": proposal.player_role_assignments,
                        "explanation": proposal.explanation,
                        "confidence": proposal.confidence,
                        "phase": "initial"
                    })
                
                # Add debate round responses
                for round_data in debate_rounds:
                    for response in round_data.agent_responses:
                        all_responses_data.append({
                            "agent_name": response.agent_name,
                            "player_role_assignments": response.player_role_assignments,
                            "explanation": response.explanation,
                            "confidence": response.confidence,
                            "phase": response.phase,
                            "agree_with": response.agree_with,
                            "disagree_with": response.disagree_with,
                            "agree_reasoning": response.agree_reasoning,
                            "disagree_reasoning": response.disagree_reasoning,
                            "round_number": round_data.round_number,
                            "player_focus": round_data.player_name
                        })
                
                self.chat_manager.add_other_agents_context(
                    agent_name, all_responses_data, "final"
                )
                
                # Get chat context for API call
                chat_context = self.chat_manager.get_chat_context_for_agent(agent_name)
                
                # Log final discussion context
                self.logger.info(f"  📝 {agent_name} FINAL DISCUSSION CHAT CONTEXT:")
                self.logger.info(f"  Total messages: {len(chat_context)}")
                
                # Make API call for final decision
                if config.provider == "openai":
                    if config.model in ["gpt-5-nano", "gpt-5", "gpt-4o"]:
                        response_obj = client.response_completion(
                            user_prompt="",
                            system_prompt="",
                            model=config.model,
                            reasoning_effort=config.reasoning_effort,
                            verbosity=config.verbosity,
                            messages=chat_context
                        )
                    else:
                        # For models that don't support reasoning, combine to single prompt
                        combined_prompt = self._combine_chat_context_to_prompt(chat_context)
                        response_obj = client.response_completion(
                            user_prompt=combined_prompt,
                            system_prompt="",
                            model=config.model
                        )
                elif config.provider == "gemini":
                    response_obj = client.chat_completion(
                        user_prompt="",
                        system_prompt="",
                        model=config.model,
                        response_schema=get_kks_response_schema_with_confidence(self.config.self_reported_confidence),
                        messages=chat_context
                    )
                else:
                    combined_prompt = self._combine_chat_context_to_prompt(chat_context)
                    response_obj = client.chat_completion(
                        user_prompt=combined_prompt,
                        system_prompt="",
                        model=config.model
                    )
                
                # Parse response
                player_assignments, explanation, confidence, _, _, _, _ = self._parse_agent_response(response_obj.text)
                
                # Add agent's final response to their chat history
                final_response_content = json.dumps({
                    "players": [{"name": name, "role": role} for name, role in player_assignments.items()],
                    "confidence": confidence,
                    "explanation": explanation
                }, indent=2)
                
                self.chat_manager.add_agent_response(
                    agent_name, final_response_content, "final"
                )
                
                fresh_votes.append({
                    "agent_name": agent_name,
                    "player_assignments": player_assignments,
                    "explanation": explanation,
                    "confidence": confidence
                })
                
                self.logger.info(f"  ✅ {agent_name} final decision completed")
                if self.config.self_reported_confidence:
                    self.logger.info(f"  📊 {agent_name} confidence: {confidence}")
                
            except Exception as e:
                self.logger.error(f"  ❌ {agent_name} final decision failed: {e}")
                # Use last known assignment if available
                last_assignment = {}
                for round_data in debate_rounds:
                    for response in round_data.agent_responses:
                        if response.agent_name == agent_name:
                            last_assignment = response.player_role_assignments
                            break
                fresh_votes.append({
                    "agent_name": agent_name,
                    "player_assignments": last_assignment,
                    "explanation": f"Error: {str(e)}",
                    "confidence": 1.0  # Low confidence for errors
                })
        
        # Conduct majority vote on fresh decisions
        final_vote = {}
        for player_name in player_names:
            votes = []
            for vote_data in fresh_votes:
                if player_name in vote_data["player_assignments"]:
                    votes.append(vote_data["player_assignments"][player_name])
            
            if votes:
                from collections import Counter
                vote_counts = Counter(votes)
                majority_role = vote_counts.most_common(1)[0][0]
                final_vote[player_name] = majority_role
                self.logger.info(f"  {player_name}: {majority_role} (fresh votes: {dict(vote_counts)})")
        
        return final_vote
    
    def _combine_chat_context_to_prompt(self, chat_context: List[Dict[str, str]]) -> str:
        """Combine chat context into a single prompt for models that don't support chat format."""
        prompt_parts = []
        
        for msg in chat_context:
            role = msg['role']
            content = msg['content']
            
            if role == "system":
                prompt_parts.append(f"SYSTEM: {content}")
            elif role == "user":
                prompt_parts.append(f"USER: {content}")
            elif role == "assistant":
                prompt_parts.append(f"ASSISTANT: {content}")
        
        return "\n\n".join(prompt_parts)
    
    def _create_debate_prompt_for_chat(self, game: ground_truth, player_name: str,
                                     current_agent_states: List[AgentResponse],
                                     previous_rounds: List[DebateRound], agent_name: str) -> str:
        """Create debate prompt for chat history format."""
        
        prompt = f"""You are participating in a debate about {player_name}'s role in this Knight-Knaves-Spy game.

Focus: We are specifically debating what role {player_name} has (knight, knave, or spy).

Your task is to:
1. Analyze the other agents' positions on {player_name}'s role
2. Decide which OTHER agents you agree with and which you disagree with
3. Provide reasoning for your agreements and disagreements
4. Make your final decision on {player_name}'s role

IMPORTANT: You are {agent_name}. Do NOT include yourself in the agree_with or disagree_with lists.

Return your response in JSON format:
{{
    "player_role": "{player_name}",
    "role": "knight/knave/spy",
    "agree_with": ["other_agent_name1", "other_agent_name2"],
    "disagree_with": ["other_agent_name3"],
    "agree_reasoning": "Brief reasoning for agreements",
    "disagree_reasoning": "Brief reasoning for disagreements"
}}"""
        
        return prompt
    
    def _create_self_adjustment_prompt_for_chat(self, game: ground_truth, player_name: str,
                                              debate_responses: List[AgentResponse],
                                              previous_rounds: List[DebateRound], 
                                              agent_name: str, agent_config: AgentConfig,
                                              current_agent_states: List[AgentResponse]) -> str:
        """Create self-adjustment prompt for chat history format."""
        
        prompt = f"""Based on the debate about {player_name}'s role, please provide your complete solution for ALL players.

You have seen the debate arguments and other agents' positions. Consider whether you want to adjust your position on {player_name} or any other players based on the arguments made.

Return your complete solution in JSON format:
{{
    "players": [
        {{"name": "player_name", "role": "role"}},
        ...
    ],
    "explanation": "Your reasoning after considering the debate"
}}"""
        
        return prompt
    
    def _create_final_discussion_prompt_for_chat(self, game: ground_truth, initial_proposals: List[AgentResponse], debate_rounds: List[DebateRound]) -> str:
        """Create final discussion prompt for chat history format."""
        
        prompt = f"""This is the FINAL DISCUSSION phase. You have seen the complete debate history.

Now make your final decision for ALL players. Consider:
1. All the arguments made during the debates
2. How your thinking may have evolved
3. Any new insights from the discussion
4. The overall consistency of the solution

Return your final solution in JSON format:
{{
    "players": [
        {{"name": "player_name", "role": "role"}},
        ...
    ],
    "explanation": "Your final comprehensive reasoning"
}}"""
        
        return prompt
    
    def _save_chat_histories(self, session: DebateSession):
        """Save chat histories for all agents."""
        if not self.chat_manager:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for agent_name, history in self.chat_manager.agent_histories.items():
            filename = f"chat_history_{agent_name}_game{session.game_id}_{timestamp}.json"
            filepath = os.path.join(self.organized_output_path, filename)
            
            # Convert to serializable format
            history_dict = {
                "agent_name": history.agent_name,
                "game_id": history.game_id,
                "messages": [
                    {
                        "role": msg.role.value,
                        "content": msg.content,
                        "timestamp": msg.timestamp,
                        "metadata": msg.metadata,
                        "phase": msg.phase,
                        "player_focus": msg.player_focus,
                        "agent_name": msg.agent_name,
                        "round_number": msg.round_number
                    }
                    for msg in history.messages
                ]
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(history_dict, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"💾 Chat history for {agent_name} saved to: {filepath}")
