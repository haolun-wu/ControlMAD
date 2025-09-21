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
from debate.chat.debate_config import DebateConfig, AgentConfig
from debate.chat.agent_chat_manager import AgentChatManager, MessageRole
from utils.utility import openai_client, gemini_client, ali_client, cstcloud, ParallelProcessor

# Data classes for the chat system
@dataclass
class AgentResponse:
    """Response from an agent in the debate system."""
    agent_name: str
    game_id: int
    round_number: int
    phase: str  # "initial", "debate", "self_adjustment", "final"
    player_role_assignments: Dict[str, str]
    explanation: str
    confidence: float
    response_obj: Any
    timestamp: str
    error: str
    agree_with: Optional[List[str]] = None
    disagree_with: Optional[List[str]] = None
    agree_reasoning: Optional[str] = None
    disagree_reasoning: Optional[str] = None

@dataclass
class DebateRound:
    """A single debate round for a specific player."""
    player_name: str
    round_number: int
    agent_responses: List[AgentResponse]
    debate_summary: str
    consensus_reached: bool
    majority_role: Optional[str]

@dataclass
class DebateSession:
    """Complete debate session for a single game."""
    game_id: int
    game_text: str
    ground_truth_solution: Dict[str, str]
    initial_proposals: List[AgentResponse]
    debate_rounds: List[DebateRound]
    final_vote: Dict[str, str]
    supervisor_decision: Optional[Dict[str, str]]
    performance_tracking: Dict[str, Any]
from prompts_chat import (
    kks_chat_system_prompt,
    kks_chat_response_schema,
    get_kks_chat_system_prompt_with_confidence,
    get_kks_chat_response_schema_with_confidence,
    get_kks_chat_debate_response_schema_with_confidence,
    get_chat_debate_prompt,
    get_chat_self_adjustment_prompt,
    get_chat_final_discussion_prompt
)


class ChatHistoryDebateSystem:
    """
    Enhanced Multi-Agent Debate System with Chat History Support.
    
    This system maintains individual chat histories for each agent, enabling
    better self-awareness through clear separation of agent messages and
    natural multi-turn dialogue format.
    """
    
    def __init__(self, debate_config: DebateConfig, secret_path: str = "secret.json", setup_logging: bool = True):
        """Initialize the chat history debate system."""
        self.config = debate_config
        self.secret_path = secret_path
        self.chat_manager: Optional[AgentChatManager] = None
        
        # Initialize agents
        self.agents = {}
        self._initialize_agents()
        
        # Initialize parallel processor with proper number of workers
        self.parallel_processor = ParallelProcessor(num_workers=debate_config.game_parallel_workers)
        
        # Setup output paths and logging (will be done per game in run_debate_session)
        if setup_logging:
            # Initialize with default game_id, will be updated per game
            self._setup_output_paths(game_id=1)
            self._setup_logging(game_id=1)
    
    def run_debate_session(self, game: ground_truth) -> DebateSession:
        """Run a complete debate for a single game with chat history support."""
        # Setup output paths and logging for this specific game
        self._setup_output_paths(game.game_id)
        self._setup_logging(game.game_id, create_log_file=True)
        
        # Ensure the log file handler is created before logging
        self._ensure_log_file_handler()
        
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
                confidence_info = f" (confidence: {response.confidence})" if self.config.self_reported_confidence and response.confidence >= 1 else ""
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
                    # All OpenAI models need combined prompt since response_completion doesn't support messages
                    combined_prompt = self._combine_chat_context_to_prompt(chat_context)
                    # Check if model supports reasoning parameters
                    if config.model in ["gpt-5-nano", "gpt-5", "gpt-4o"]:
                        response_obj = client.response_completion(
                            user_prompt=combined_prompt,
                            system_prompt="",
                            model=config.model,
                            reasoning_effort=config.reasoning_effort,
                            verbosity=config.verbosity
                        )
                    else:
                        # For models that don't support reasoning, use basic response completion
                        response_obj = client.response_completion(
                            user_prompt=combined_prompt,
                            system_prompt="",
                            model=config.model
                        )
                elif config.provider == "gemini":
                    # Gemini doesn't support messages parameter, so combine to single prompt
                    combined_prompt = self._combine_chat_context_to_prompt(chat_context)
                    response_obj = client.chat_completion(
                        user_prompt=combined_prompt,
                        system_prompt="",
                        model=config.model,
                        response_schema=get_kks_chat_response_schema_with_confidence(self.config.self_reported_confidence)
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
                    response_obj=None,  # No response obj for errors
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
                    # All OpenAI models need combined prompt since response_completion doesn't support messages
                    combined_prompt = self._combine_chat_context_to_prompt(chat_context)
                    # Check if model supports reasoning parameters
                    if config.model in ["gpt-5-nano", "gpt-5", "gpt-4o"]:
                        response_obj = client.response_completion(
                            user_prompt=combined_prompt,
                            system_prompt="",
                            model=config.model,
                            reasoning_effort=config.reasoning_effort,
                            verbosity=config.verbosity
                        )
                    else:
                        # For models that don't support reasoning, use basic response completion
                        response_obj = client.response_completion(
                            user_prompt=combined_prompt,
                            system_prompt="",
                            model=config.model
                        )
                elif config.provider == "gemini":
                    # Use debate-specific schema for Gemini
                    combined_prompt = self._combine_chat_context_to_prompt(chat_context)
                    debate_response_schema = get_kks_chat_debate_response_schema_with_confidence(self.config.self_reported_confidence)
                    response_obj = client.chat_completion(
                        user_prompt=combined_prompt,
                        system_prompt="",
                        model=config.model,
                        response_schema=debate_response_schema
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
                    response_obj=None,  # No response obj for errors
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
                
                # Log the debate responses data being added
                self.logger.info(f"      🗣️ ADDING DEBATE RESPONSES TO {agent_name}'s CHAT HISTORY:")
                for i, response_data in enumerate(debate_responses_data):
                    self.logger.info(f"        Debate Response {i+1}:")
                    self.logger.info(f"          Agent: {response_data['agent_name']}")
                    self.logger.info(f"          Assignments: {response_data['player_role_assignments']}")
                    self.logger.info(f"          Agrees with: {response_data['agree_with']}")
                    self.logger.info(f"          Disagrees with: {response_data['disagree_with']}")
                    self.logger.info(f"          Agree reasoning: {response_data['agree_reasoning']}")
                    self.logger.info(f"          Disagree reasoning: {response_data['disagree_reasoning']}")
                    self.logger.info(f"          Confidence: {response_data['confidence']}")
                    self.logger.info("")
                
                self.chat_manager.add_other_agents_context(
                    agent_name, debate_responses_data, "self_adjustment", round_num
                )
                
                # Get chat context for API call
                chat_context = self.chat_manager.get_chat_context_for_agent(agent_name)
                
                # Log self-adjustment context with detailed chat history
                self.logger.info(f"      📝 {agent_name} SELF-ADJUSTMENT CHAT CONTEXT for {player_name}:")
                self.logger.info(f"      Total messages: {len(chat_context)}")
                
                # Log each message in the chat context to show debate reasoning
                for i, message in enumerate(chat_context):
                    self.logger.info(f"      Message {i+1} ({message['role']}):")
                    content = message['content']
                    if len(content) > 500:
                        self.logger.info(f"        {content[:500]}...")
                        self.logger.info(f"        [TRUNCATED - Full length: {len(content)} chars]")
                    else:
                        self.logger.info(f"        {content}")
                    self.logger.info("")
                
                # Log the self-adjustment prompt that gets added
                self.logger.info(f"      🔄 SELF-ADJUSTMENT PROMPT for {agent_name}:")
                self.logger.info(f"      {adjustment_prompt}")
                self.logger.info("")
                
                # Make API call
                if config.provider == "openai":
                    # All OpenAI models need combined prompt since response_completion doesn't support messages
                    combined_prompt = self._combine_chat_context_to_prompt(chat_context)
                    
                    # Log the final combined prompt that gets sent to the API
                    self.logger.info(f"      📤 FINAL COMBINED PROMPT SENT TO {agent_name} API:")
                    if len(combined_prompt) > 1000:
                        self.logger.info(f"        {combined_prompt[:1000]}...")
                        self.logger.info(f"        [TRUNCATED - Full length: {len(combined_prompt)} chars]")
                    else:
                        self.logger.info(f"        {combined_prompt}")
                    self.logger.info("")
                    
                    # Check if model supports reasoning parameters
                    if config.model in ["gpt-5-nano", "gpt-5", "gpt-4o"]:
                        response_obj = client.response_completion(
                            user_prompt=combined_prompt,
                            system_prompt="",
                            model=config.model,
                            reasoning_effort=config.reasoning_effort,
                            verbosity=config.verbosity
                        )
                    else:
                        # For models that don't support reasoning, use basic response completion
                        response_obj = client.response_completion(
                            user_prompt=combined_prompt,
                            system_prompt="",
                            model=config.model
                        )
                elif config.provider == "gemini":
                    combined_prompt = self._combine_chat_context_to_prompt(chat_context)
                    response_obj = client.chat_completion(
                        user_prompt=combined_prompt,
                        system_prompt="",
                        model=config.model,
                        response_schema=get_kks_chat_response_schema_with_confidence(self.config.self_reported_confidence)
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
                    response_obj=None,  # No response obj for errors
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
                        # All OpenAI models need combined prompt since response_completion doesn't support messages
                        combined_prompt = self._combine_chat_context_to_prompt(chat_context)
                        response_obj = client.response_completion(
                            user_prompt=combined_prompt,
                            system_prompt="",
                            model=config.model,
                            reasoning_effort=config.reasoning_effort,
                            verbosity=config.verbosity
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
                    combined_prompt = self._combine_chat_context_to_prompt(chat_context)
                    response_obj = client.chat_completion(
                        user_prompt=combined_prompt,
                        system_prompt="",
                        model=config.model,
                        response_schema=get_kks_chat_response_schema_with_confidence(self.config.self_reported_confidence)
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

GAME INFORMATION:
{game.text_game}

CURRENT FOCUS: We are debating the role of {player_name}.

You can see the conversation history above, which includes:
- Your own previous responses (marked as your messages)
- Other agents' positions and reasoning
- The debate context and instructions

Your task is to:
1. Analyze the other agents' positions on {player_name}'s role from the conversation history
2. Decide which OTHER agents you agree with and which you disagree with
3. Provide reasoning for your agreements and disagreements
4. Make your final decision on {player_name}'s role

Note: You can see your own previous responses in the conversation history, so you have natural self-awareness of your own position.

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

GAME INFORMATION:
{game.text_game}

CURRENT FOCUS: We just finished debating {player_name}'s role.

You can see the conversation history above, which includes:
- Your own previous responses and solutions
- Other agents' positions and reasoning
- The debate arguments and agreements/disagreements

Based on the debate, please provide your final assessment. You may adjust your position on {player_name} or any other players if you've been convinced by the arguments.

You can reference your own previous responses and those of other agents from the conversation history.

IMPORTANT: For this self-adjustment phase, provide your complete solution for ALL players, not just {player_name}.

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
        
        prompt = f"""This is the FINAL DISCUSSION phase. You have seen the complete debate history in the conversation.

GAME INFORMATION:
{game.text_game}

You can see the conversation history above, which includes:
- Your own responses throughout all phases (initial, debate, self-adjustment)
- Other agents' positions and reasoning from all phases
- The complete debate history and evolution of arguments

Now make your final decision for ALL players. Consider:
1. All the arguments made during the debates (visible in conversation history)
2. How your thinking may have evolved through the conversation
3. Any new insights from the discussion
4. The overall consistency of the solution

You can reference the entire conversation history including your own responses and those of other agents.

Return your final solution in JSON format:
{{
    "players": [
        {{"name": "player_name", "role": "role"}},
        ...
    ],
    "explanation": "Your final comprehensive reasoning"
}}"""
        
        return prompt
    
    def _create_supervisor_prompt_for_chat(self, game: ground_truth, initial_proposals: List[AgentResponse], debate_rounds: List[DebateRound]) -> str:
        """Create supervisor prompt for chat history format."""
        
        prompt = f"""You are a supervisor AI tasked with making the final decision in a multi-agent debate about a Knight-Knaves-Spy game.

GAME INFORMATION:
{game.text_game}

COMPLETE DEBATE HISTORY:

INITIAL PROPOSALS:
"""
        
        for proposal in initial_proposals:
            prompt += f"\n{proposal.agent_name} initially proposed: {proposal.player_role_assignments}"
            prompt += f"\nTheir reasoning: {proposal.explanation[:200]}...\n"
        
        prompt += "\nDEBATE ROUNDS:\n"
        for round_data in debate_rounds:
            prompt += f"\n--- Round {round_data.round_number}: {round_data.player_name} ---\n"
            for response in round_data.agent_responses:
                role = response.player_role_assignments.get(round_data.player_name, "unknown")
                prompt += f"{response.agent_name} thought {round_data.player_name} is a {role}\n"
                
                # Include agreement/disagreement information if available (for debate phase responses)
                if response.phase == "debate":
                    if response.agree_with and len(response.agree_with) > 0:
                        prompt += f"  Agreed with: {', '.join(response.agree_with)}"
                        if response.agree_reasoning and response.agree_reasoning.strip():
                            prompt += f" - Reasoning: {response.agree_reasoning[:150]}..."
                        prompt += "\n"
                    
                    if response.disagree_with and len(response.disagree_with) > 0:
                        prompt += f"  Disagreed with: {', '.join(response.disagree_with)}"
                        if response.disagree_reasoning and response.disagree_reasoning.strip():
                            prompt += f" - Reasoning: {response.disagree_reasoning[:150]}..."
                        prompt += "\n"
                else:
                    # Include explanation for non-debate phases (initial, self_adjustment)
                    if response.explanation:
                        prompt += f"Reasoning: {response.explanation[:200]}...\n"
        
        prompt += """

SUPERVISOR INSTRUCTIONS:
As the supervisor, you have access to the complete debate history. The agents have been unable to reach consensus, so you must make the final decision. Consider:

1. All initial proposals and their reasoning
2. The evolution of arguments through the debate rounds
3. The consistency and logic of each agent's reasoning
4. The overall coherence of the solution
5. Any patterns or insights that emerged during the debate

Make your decision based on the most logical and well-reasoned arguments you observed.

Return your response in the same JSON format:
{
    "players": [
        {"name": "player_name", "role": "role"},
        ...
    ],
    "explanation": "Your final decision with comprehensive reasoning based on the complete debate history"
}

IMPORTANT: Keep your explanation having details but less than 100 words."""
        
        return prompt
    
    def _get_supervisor_decision(self, game: ground_truth, initial_proposals: List[AgentResponse], debate_rounds: List[DebateRound]) -> Dict[str, str]:
        """Get supervisor decision when consensus cannot be reached (chat history version)."""
        self.logger.info("Getting supervisor decision...")
        
        try:
            # Initialize supervisor client
            if self.config.supervisor_provider == "openai":
                supervisor_client = openai_client(self.secret_path)
            elif self.config.supervisor_provider == "gemini":
                supervisor_client = gemini_client(self.secret_path)
            elif self.config.supervisor_provider == "ali":
                supervisor_client = ali_client(self.secret_path)
            else:
                supervisor_client = cstcloud(self.secret_path)
            
            # Create supervisor prompt using chat history version
            supervisor_prompt = self._create_supervisor_prompt_for_chat(game, initial_proposals, debate_rounds)
            
            # Make API call
            system_prompt = get_kks_chat_system_prompt_with_confidence(game.num_player, self.config.self_reported_confidence)
            response_schema = get_kks_chat_response_schema_with_confidence(self.config.self_reported_confidence)
            
            if self.config.supervisor_provider == "openai":
                # Check if supervisor model supports reasoning parameters
                if self.config.supervisor_model in ["gpt-5-nano", "gpt-5", "gpt-4o"]:
                    response_obj = supervisor_client.response_completion(
                        user_prompt=supervisor_prompt,
                        system_prompt=system_prompt,
                        model=self.config.supervisor_model,
                        reasoning_effort="high",
                        verbosity="high"
                    )
                else:
                    # Use regular response completion for models that don't support reasoning
                    response_obj = supervisor_client.response_completion(
                        user_prompt=supervisor_prompt,
                        system_prompt=system_prompt,
                        model=self.config.supervisor_model
                    )
            elif self.config.supervisor_provider == "gemini":
                response_obj = supervisor_client.chat_completion(
                    user_prompt=supervisor_prompt,
                    system_prompt=system_prompt,
                    model=self.config.supervisor_model,
                    response_schema=response_schema
                )
            else:
                response_obj = supervisor_client.chat_completion(
                    user_prompt=supervisor_prompt,
                    system_prompt=system_prompt,
                    model=self.config.supervisor_model
                )
            
            # Parse supervisor response
            player_assignments, explanation, confidence, _, _, _, _ = self._parse_agent_response(response_obj.text)
            self.logger.info(f"Supervisor decision: {player_assignments}")
            if self.config.self_reported_confidence:
                self.logger.info(f"Supervisor confidence: {confidence}")
            else:
                self.logger.info(f"Supervisor confidence: not requested (SRC disabled)")
            
            return player_assignments
            
        except Exception as e:
            self.logger.error(f"Error getting supervisor decision: {e}")
            return {}
    
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
    
    # Missing methods that were inherited from MultiAgentDebateSystem
    
    def _initialize_agents(self):
        """Initialize agents based on the debate configuration."""
        for agent_config in self.config.agents:
            agent_name = agent_config.name
            
            # Initialize client based on provider
            if agent_config.provider == "openai":
                client = openai_client(self.secret_path)
            elif agent_config.provider == "gemini":
                client = gemini_client(self.secret_path)
            elif agent_config.provider == "ali":
                client = ali_client(self.secret_path)
            elif agent_config.provider == "cst":
                client = cstcloud(self.secret_path)
            else:
                raise ValueError(f"Unknown provider: {agent_config.provider}")
            
            self.agents[agent_name] = {
                "config": agent_config,
                "client": client
            }
    
    def _setup_logging(self, game_id: int = 1, create_log_file: bool = True):
        """Setup logging for the debate system (borrowed from traditional system)."""
        import logging
        from datetime import datetime
        
        # Only setup logging once per game to avoid multiple log files
        if hasattr(self, 'logger') and self.logger is not None and hasattr(self, 'current_game_id') and self.current_game_id == game_id:
            return
        
        # Use a more unique timestamp with microseconds to avoid conflicts in parallel processing
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        log_filename = f"debate_log_game{game_id}_{timestamp}.log"
        log_path = os.path.join(self.organized_output_path, log_filename)
        
        # Create logger with game-specific name
        logger_name = f"chat_debate_system_game{game_id}_{timestamp}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Only create file handler if requested
        if create_log_file:
            # Store the log path for later use
            self.log_file_path = log_path
            # Don't create the file handler yet - we'll create it when we have content to log
        
        # Always log to console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Track current game ID to avoid duplicate setup
        self.current_game_id = game_id
    
    def _ensure_log_file_handler(self):
        """Ensure the log file handler is created (borrowed from traditional system)."""
        import logging
        
        # Check if we already have a file handler
        has_file_handler = any(isinstance(handler, logging.FileHandler) for handler in self.logger.handlers)
        
        if not has_file_handler and hasattr(self, 'log_file_path'):
            # Create file handler
            file_handler = logging.FileHandler(self.log_file_path, encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def _setup_output_paths(self, game_id: int = 1):
        """Setup output paths for results using the proper folder structure."""
        # Use the DebateConfig's method to get the organized output path
        self.organized_output_path = self.config.get_organized_output_path(game_id)
        
        # Create directory if it doesn't exist
        os.makedirs(self.organized_output_path, exist_ok=True)
    
    def _parse_confidence(self, confidence_value) -> float:
        """Parse confidence value and ensure it's in the 1-10 range."""
        try:
            confidence = float(confidence_value)
            # Ensure confidence is in valid range (1-10)
            if confidence < 1.0:
                confidence = 1.0
            elif confidence > 10.0:
                confidence = 10.0
            return confidence
        except (ValueError, TypeError):
            # If parsing fails, return default medium confidence
            return 5.0
    
    def _parse_ground_truth_solution(self, game: ground_truth) -> Dict[str, str]:
        """Parse ground truth solution from game data."""
        try:
            solution_text = game.text_solution.strip()
            import re
            
            pairs = []
            pattern = r'(\w+)\s+is\s+a\s+(knight|knave|spy)\.'
            matches = re.findall(pattern, solution_text, re.IGNORECASE)
            
            for player_name, role in matches:
                pairs.append((player_name, role.lower()))
            
            return dict(pairs)
            
        except Exception as e:
            self.logger.error(f"Error parsing ground truth solution: {e}")
            self.logger.error(f"Solution text: {game.text_solution}")
            return {}
    
    def _parse_agent_response(self, response_text: str, phase: str = "initial") -> Tuple[Dict[str, str], str, float, Optional[List[str]], Optional[List[str]], Optional[str], Optional[str]]:
        """Parse agent response to extract player assignments, explanation, confidence, and debate analysis."""
        try:
            # First try to extract JSON from the response (handle extra text after JSON)
            json_text = self._extract_json_from_response(response_text)
            response_data = json.loads(json_text)
            
            # Extract player assignments (borrowed from traditional system)
            player_assignments = {}
            if "players" in response_data and isinstance(response_data["players"], list):
                # New format with players array
                for player_info in response_data["players"]:
                    if isinstance(player_info, dict) and "name" in player_info and "role" in player_info:
                        name = player_info.get("name", "")
                        role = player_info.get("role", "")
                        if name and role:
                            player_assignments[name] = role
            else:
                # Old format - direct name: role mapping
                for key, value in response_data.items():
                    if key not in ["explanation", "confidence", "agree_with", "disagree_with", "agree_reasoning", "disagree_reasoning"] and isinstance(value, str):
                        player_assignments[key] = value
            
            # Extract explanation
            explanation = response_data.get("explanation", "")
            
            # Extract confidence using the same method as traditional system
            confidence = self._parse_confidence(response_data.get("confidence", 5))
            
            # Extract debate-specific fields
            agree_with = None
            disagree_with = None
            agree_reasoning = None
            disagree_reasoning = None
            
            if phase == "debate":
                agree_with = response_data.get("agree_with", [])
                disagree_with = response_data.get("disagree_with", [])
                agree_reasoning = response_data.get("agree_reasoning", "")
                disagree_reasoning = response_data.get("disagree_reasoning", "")
            
            return player_assignments, explanation, confidence, agree_with, disagree_with, agree_reasoning, disagree_reasoning
            
        except Exception as e:
            self.logger.error(f"Error parsing agent response: {e}")
            self.logger.error(f"Raw response: {response_text}")
            # Return default values instead of empty dict to avoid {} in logs
            return {"error": "parsing_failed"}, f"Error parsing response: {str(e)}", 5.0, None, None, None, None
    
    def _extract_json_from_response(self, response_text: str) -> str:
        """Extract JSON from response text, handling extra text after JSON (borrowed from traditional system)."""
        # Try to find JSON object boundaries
        # Look for opening brace and try to find matching closing brace
        start_idx = response_text.find('{')
        if start_idx == -1:
            raise ValueError("No JSON object found")
        
        # Count braces to find the end of the JSON object
        brace_count = 0
        end_idx = start_idx
        
        for i, char in enumerate(response_text[start_idx:], start_idx):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
        
        if brace_count != 0:
            raise ValueError("Unmatched braces in JSON")
        
        return response_text[start_idx:end_idx]
    
    def _create_debate_summary(self, responses: List[AgentResponse], player_name: str) -> str:
        """Create a summary of debate responses for a specific player."""
        if not responses:
            return "No responses received."
        
        summary_parts = []
        for response in responses:
            role = response.player_role_assignments.get(player_name, "unknown")
            summary_parts.append(f"{response.agent_name}: {role}")
        
        return "; ".join(summary_parts)
    
    def _check_consensus(self, responses: List[AgentResponse], player_name: str) -> Optional[str]:
        """Check if there's consensus among agents for a specific player's role."""
        if not responses:
            return None
        
        # Count votes for each role
        role_counts = {}
        for response in responses:
            role = response.player_role_assignments.get(player_name, "unknown")
            if role != "unknown":
                role_counts[role] = role_counts.get(role, 0) + 1
        
        if not role_counts:
            return None
        
        # Find the role with the most votes
        majority_role = max(role_counts.items(), key=lambda x: x[1])[0]
        majority_count = role_counts[majority_role]
        
        # Check if it's a true majority (more than half)
        total_votes = len(responses)
        if majority_count > total_votes / 2:
            return majority_role
        
        return None
    
    def _is_consensus_reached(self, final_vote: Dict[str, str]) -> bool:
        """Check if consensus is reached in the final vote."""
        if not final_vote:
            return False
        
        # Simple consensus check - if we have a vote for each player, consider it consensus
        return len(final_vote) > 0
    
    def _create_performance_tracking(self, game: ground_truth, initial_proposals: List[AgentResponse], 
                                   debate_rounds: List[DebateRound], final_vote: Dict[str, str], 
                                   supervisor_decision: Optional[Dict[str, str]]) -> Dict[str, Any]:
        """Create performance tracking data for the debate session."""
        gt_solution = self._parse_ground_truth_solution(game)
        
        # Calculate accuracy for initial proposals
        initial_accuracy = {}
        for proposal in initial_proposals:
            correct = 0
            total = len(gt_solution)
            for player, gt_role in gt_solution.items():
                if proposal.player_role_assignments.get(player) == gt_role:
                    correct += 1
            initial_accuracy[proposal.agent_name] = correct / total if total > 0 else 0
        
        # Calculate final accuracy
        final_solution = supervisor_decision if supervisor_decision else final_vote
        final_correct = 0
        final_total = len(gt_solution)
        for player, gt_role in gt_solution.items():
            if final_solution.get(player) == gt_role:
                final_correct += 1
        final_accuracy = final_correct / final_total if final_total > 0 else 0
        
        return {
            "initial_accuracy": initial_accuracy,
            "final_accuracy": final_accuracy,
            "ground_truth": gt_solution,
            "final_solution": final_solution,
            "supervisor_used": supervisor_decision is not None
        }
    
    def _save_debate_session(self, session: DebateSession):
        """Save debate session to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"debate_session_game{session.game_id}_{timestamp}.json"
        filepath = os.path.join(self.organized_output_path, filename)
        
        # Convert session to dictionary
        session_dict = asdict(session)
        
        # Convert datetime objects to strings
        for proposal in session_dict["initial_proposals"]:
            if "timestamp" in proposal:
                proposal["timestamp"] = proposal["timestamp"]
        
        for round_data in session_dict["debate_rounds"]:
            for response in round_data["agent_responses"]:
                if "timestamp" in response:
                    response["timestamp"] = response["timestamp"]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(session_dict, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"💾 Debate session saved to: {filepath}")
    
    def _generate_visualizations(self, session: DebateSession):
        """Generate visualizations for the debate session."""
        # This is a placeholder - you can implement visualization generation here
        self.logger.info("📊 Visualization generation placeholder - implement as needed")
    
    def run_batch_debate(self, games: List[ground_truth]) -> List[DebateSession]:
        """Run debates for multiple games sequentially."""
        sessions = []
        
        for i, game in enumerate(games, 1):
            self.logger.info(f"\n🎮 Starting Game {i}/{len(games)}: Game {game.game_id}")
            try:
                session = self.run_debate_session(game)
                sessions.append(session)
                self.logger.info(f"✅ Completed Game {game.game_id}")
            except Exception as e:
                self.logger.error(f"❌ Failed Game {game.game_id}: {e}")
                # Create a minimal session for failed games
                failed_session = DebateSession(
                    game_id=game.game_id,
                    game_text=game.text_game,
                    ground_truth_solution={},
                    initial_proposals=[],
                    debate_rounds=[],
                    final_vote={},
                    supervisor_decision=None,
                    performance_tracking={"error": str(e)}
                )
                sessions.append(failed_session)
        
        return sessions
    
    def run_parallel_batch_debate(self, games: List[ground_truth]) -> List[DebateSession]:
        """Run debates for multiple games in parallel."""
        
        # Use parallel processor to run all games (borrowed from traditional system)
        print(f"🚀 Starting parallel processing of {len(games)} games")
        print(f"🔧 Using {self.parallel_processor.num_workers} workers for game-level parallelism")
        
        # Create a wrapper function that handles logging setup for each game
        def run_game_with_logging(game: ground_truth) -> DebateSession:
            # Create a temporary system instance for this specific game without logging
            temp_system = ChatHistoryDebateSystem(self.config, self.secret_path, setup_logging=False)
            # Use the same parallel processor (shared resources)
            temp_system.parallel_processor = self.parallel_processor
            
            # Setup logging for this specific game
            temp_system._setup_output_paths(game.game_id)
            temp_system._setup_logging(game.game_id, create_log_file=True)
            temp_system._ensure_log_file_handler()
            
            # Log the parallel processing message
            temp_system.logger.info(f"\n{'='*60}")
            temp_system.logger.info(f"Running parallel debate for Game {game.game_id}")
            temp_system.logger.info(f"{'='*60}")
            
            # Run the debate session
            session = temp_system.run_debate_session(game)
            temp_system.logger.info(f"✅ Completed Game {game.game_id}")
            return session
        
        sessions = self.parallel_processor.process_tasks(run_game_with_logging, games, preserve_order=True)
        
        # Filter out None results
        sessions = [s for s in sessions if s is not None]
        
        print(f"🏁 Completed parallel batch debate: {len(sessions)} sessions")
        return sessions
