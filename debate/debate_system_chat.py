"""
Enhanced Multi-Agent Debate System with Chat History

This module extends the original debate system to use chat histories for better
agent self-awareness and natural multi-turn dialogue format.
"""

import json
import os
import copy

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
from debate.agent_chat_manager import AgentChatManager, MessageRole
from utils.utility import (
    openai_client,
    gemini_client,
    ali_client,
    cstcloud,
    ParallelProcessor,
)
from utils.logging_utils import HTMLFileHandler


from debate.prompts_chat import (
    kks_chat_system_prompt,
    kks_chat_response_schema,
    get_kks_chat_system_prompt_with_confidence,
    get_kks_chat_response_schema_with_confidence,
    get_kks_chat_debate_response_schema_with_confidence,
    get_kks_chat_self_adjustment_response_schema_with_confidence,
    get_debate_order_selection_prompt,
    get_debate_order_response_schema,
    get_chat_debate_prompt_for_chat,
    get_chat_self_adjustment_prompt_for_chat,
    get_chat_final_discussion_prompt_for_chat,
    get_chat_supervisor_prompt_for_chat,
    generate_kks_chat_self_adjustment_response_schema,
    generate_kks_chat_debate_response_schema,
    get_kks_chat_debate_response_schema_dynamic_with_confidence
)


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


class ChatHistoryDebateSystem:
    """
    Enhanced Multi-Agent Debate System with Chat History Support.

    This system maintains individual chat histories for each agent, enabling
    better self-awareness through clear separation of agent messages and
    natural multi-turn dialogue format.
    """

    def __init__(
        self,
        debate_config: DebateConfig,
        secret_path: str = "secret.json",
        setup_logging: bool = True,
    ):
        """Initialize the chat history debate system."""
        self.config = debate_config
        self.secret_path = secret_path
        self.chat_manager: Optional[AgentChatManager] = None

        # Initialize agents
        self.agents = {}
        self._initialize_agents()

        # Initialize parallel processor with proper number of workers
        self.parallel_processor = ParallelProcessor(
            num_workers=debate_config.game_parallel_workers
        )

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
        print(
            f"📁 Current log: {self.organized_output_path}/debate_log_game{game.game_id}_*.log"
        )
        print("=" * 60)

        self.logger.info(f"🎯 Starting chat history debate for Game {game.game_id}")
        self.logger.info(
            f"Players: {[f'Player {i+1}' for i in range(game.num_player)]}"
        )

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
            self.logger.info(
                f"  {proposal.agent_name}: {proposal.player_role_assignments}"
            )
            if proposal.error:
                self.logger.error(f"    ❌ Error: {proposal.error}")
            else:
                self.logger.info(f"    ✅ Success")

            # Log per-player accuracy for initial proposals
            self.logger.info(
                f"    📊 {proposal.agent_name} Initial Accuracy by Player:"
            )
            for player, gt_role in gt_solution.items():
                predicted_role = proposal.player_role_assignments.get(player, "unknown")
                correct = "✅" if predicted_role == gt_role else "❌"
                self.logger.info(
                    f"      {player}: {predicted_role} vs {gt_role} {correct}"
                )

        # Phase 2: Debate rounds for each player with chat history
        self.logger.info("🗣️ Phase 2: Debate Rounds (Chat History)")
        debate_rounds = []

        # Select debate order based on configuration
        player_names = self._select_debate_order(game, initial_proposals)
        
        # Generate dynamic schema with specific player names for this game
        # Note: Schema is now generated locally within each function call to avoid race conditions
        self.logger.info(f"📋 Will generate dynamic schemas locally for players: {player_names}")

        # Track the current state of all agents (starts with initial proposals)
        current_agent_states = initial_proposals

        for player_name in player_names:
            self.logger.info(f"--- Debating {player_name}'s role ---")
            self.logger.info(
                f"🎯 Ground Truth: {player_name} is a {gt_solution.get(player_name, 'unknown')}"
            )

            # Run multiple debate rounds for this player based on depth
            player_debate_rounds = (
                []
            )  # Track all debate rounds for this specific player
            all_debate_responses = (
                []
            )  # Collect all debate responses across depth rounds

            # Phase 1: Run all debate rounds (no self-adjustment between rounds)
            for depth_round in range(1, self.config.depth + 1):
                self.logger.info(
                    f"🔄 Depth Round {depth_round}/{self.config.depth} for {player_name} (Debate Only)"
                )

                # Run only the debate phase, no self-adjustment
                debate_responses = self._conduct_debate_for_player_with_chat(
                    game,
                    player_name,
                    len(debate_rounds) + 1,
                    current_agent_states,
                    debate_rounds,
                )

                # Store debate responses for later self-adjustment
                all_debate_responses.extend(debate_responses)

                # Create a debate round object (without self-adjustment responses)
                debate_round = DebateRound(
                    player_name=player_name,
                    round_number=len(debate_rounds) + 1,
                    agent_responses=debate_responses,  # Only debate responses
                    debate_summary=self._create_debate_summary(
                        debate_responses, player_name
                    ),
                    consensus_reached=False,  # Will be determined after self-adjustment
                    majority_role=None,
                )

                debate_rounds.append(debate_round)
                player_debate_rounds.append(debate_round)

                # Update current agent states to the debate results from this round
                current_agent_states = debate_responses

                # Show round results
                self.logger.info(
                    f"📊 Round {debate_round.round_number} Results for {player_name} (Depth {depth_round} - Debate):"
                )
                for response in debate_round.agent_responses:
                    # Handle different response formats based on phase
                    if response.phase == "debate":
                        role = response.player_role_assignments.get("role", "unknown")
                    else:
                        role = response.player_role_assignments.get(
                            player_name, "unknown"
                        )
                    correct = "✅" if role == gt_solution.get(player_name) else "❌"
                    confidence_info = (
                        f" (confidence: {response.confidence})"
                        if self.config.self_reported_confidence
                        and response.confidence >= 1
                        else ""
                    )
                    self.logger.info(
                        f"  {response.agent_name}: {role} {correct}{confidence_info}"
                    )

                    # Log detailed per-player accuracy for this round
                    # For debate phase, only log the player being debated since that's the only role the agent provided
                    if response.phase == "debate":
                        self.logger.info(
                            f"    📊 {response.agent_name} Round {debate_round.round_number} Accuracy for {player_name}:"
                        )
                        predicted_role = response.player_role_assignments.get(
                            "role", "unknown"
                        )
                        gt_role = gt_solution.get(player_name, "unknown")
                        correct = "✅" if predicted_role == gt_role else "❌"
                        self.logger.info(
                            f"      {player_name}: {predicted_role} vs {gt_role} {correct}"
                        )
                    else:
                        # For other phases, log all players
                        self.logger.info(
                            f"    📊 {response.agent_name} Round {debate_round.round_number} Accuracy by Player:"
                        )
                        for player, gt_role in gt_solution.items():
                            predicted_role = response.player_role_assignments.get(
                                player, "unknown"
                            )
                            correct = "✅" if predicted_role == gt_role else "❌"
                            self.logger.info(
                                f"      {player}: {predicted_role} vs {gt_role} {correct}"
                            )
                self.logger.info(
                    f"🎯 Depth Round {depth_round} debate completed for {player_name}"
                )

            # Phase 2: Single self-adjustment using ALL debate responses from all depth rounds
            self.logger.info(
                f"🔄 Self-adjustment for {player_name} using {len(all_debate_responses)} debate responses from {self.config.depth} depth rounds"
            )

            if self.config.enable_self_adjustment:
                adjusted_responses = self._conduct_self_adjustment_with_chat(
                    game,
                    player_name,
                    len(debate_rounds),
                    all_debate_responses,
                    debate_rounds,
                    current_agent_states,
                    initial_proposals,
                )

                # Update the final debate round with self-adjustment responses
                final_round = debate_rounds[-1]  # Get the last round
                final_round.agent_responses.extend(
                    adjusted_responses
                )  # Add self-adjustment responses

                # Update current agent states to the self-adjustment results
                current_agent_states = adjusted_responses

                # Show self-adjustment results
                self.logger.info(f"📊 Self-adjustment Results for {player_name}:")
                for response in adjusted_responses:
                    # Self-adjustment responses should have full player assignments, not just the debated player
                    role = response.player_role_assignments.get(player_name, "unknown")
                    correct = "✅" if role == gt_solution.get(player_name) else "❌"
                    confidence_info = (
                        f" (confidence: {response.confidence})"
                        if self.config.self_reported_confidence
                        and response.confidence >= 1
                        else ""
                    )
                    self.logger.info(
                        f"  {response.agent_name}: {role} {correct}{confidence_info}"
                    )

                    # Log detailed per-player accuracy for self-adjustment
                    self.logger.info(
                        f"    📊 {response.agent_name} Self-adjustment Accuracy by Player:"
                    )
                    for player, gt_role in gt_solution.items():
                        predicted_role = response.player_role_assignments.get(
                            player, "unknown"
                        )
                        correct = "✅" if predicted_role == gt_role else "❌"
                        self.logger.info(
                            f"      {player}: {predicted_role} vs {gt_role} {correct}"
                        )

                # Check for consensus in self-adjustment responses
                consensus_role = self._check_consensus(adjusted_responses, player_name)
                final_round.consensus_reached = consensus_role is not None
                final_round.majority_role = consensus_role

                if consensus_role:
                    self.logger.info(
                        f"🎯 CONSENSUS REACHED for {player_name}: {consensus_role}"
                    )
                else:
                    self.logger.info(
                        f"❌ NO CONSENSUS for {player_name} - agents disagree"
                    )
            else:
                # No self-adjustment, just use the last debate responses
                current_agent_states = all_debate_responses[
                    -len(self.agents) :
                ]  # Last responses from each agent

            self.logger.info(
                f"✅ All {self.config.depth} depth rounds + self-adjustment completed for {player_name}"
            )

        # Phase 3: Final Discussion and Fresh Voting with chat history
        self.logger.info("🗣️ Phase 3: Final Discussion and Fresh Voting (Chat History)")
        self.logger.info("=" * 40)
        final_vote = self._conduct_final_discussion_and_vote_with_chat(
            game, initial_proposals, debate_rounds
        )

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
            supervisor_decision = self._get_supervisor_decision(
                game, initial_proposals, debate_rounds
            )

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
            performance_tracking=performance_tracking,
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

    def _get_initial_proposals_with_chat(
        self, game: ground_truth
    ) -> List[AgentResponse]:
        """Get initial proposals from all agents using chat history."""

        def get_single_proposal_with_chat(agent_name: str) -> AgentResponse:
            agent_info = self.agents[agent_name]
            client = agent_info["client"]
            config = agent_info["config"]

            self.logger.info(
                f"    🤖 {agent_name} ({config.provider}/{config.model}) getting initial proposal with chat history..."
            )

            try:
                # Set up initial chat history for this agent
                self.chat_manager.create_initial_system_message(
                    agent_name,
                    game.text_game,
                    game.num_player,
                    self.config.self_reported_confidence,
                )
                self.chat_manager.create_game_context_message(
                    agent_name, game.text_game
                )

                # Get chat context for API call
                chat_context = self.chat_manager.get_chat_context_for_agent(agent_name)

                # Combine chat context into a single prompt for logging and API call
                combined_prompt = self._combine_chat_context_to_prompt(chat_context)

                # Log the complete initial proposal prompt with all context (RED TEXT)
                self.logger.info(
                    f"    🎯 COMPLETE INITIAL PROPOSAL PROMPT for {agent_name}:"
                )
                self.logger.info(
                    f"[INITIAL PROPOSAL PROMPT - RED] \033[31m{combined_prompt}\033[0m"
                )
                self.logger.info("")

                # Log prompt length for debugging
                self.logger.info(
                    f"    📊 INITIAL PROPOSAL PROMPT LENGTH: {len(combined_prompt)} characters"
                )
                self.logger.info("")

                # Make API call based on provider
                if config.provider == "openai":
                    # Check if model supports reasoning parameters
                    if config.model in ["gpt-5-nano", "gpt-5", "gpt-4o"]:
                        response_obj = client.response_completion(
                            user_prompt=combined_prompt,
                            system_prompt="",
                            model=config.model,
                            reasoning_effort=config.reasoning_effort,
                            verbosity=config.verbosity,
                        )
                    else:
                        # For models that don't support reasoning, use basic response completion
                        response_obj = client.response_completion(
                            user_prompt=combined_prompt,
                            system_prompt="",
                            model=config.model,
                        )
                elif config.provider == "gemini":
                    # Gemini doesn't support messages parameter, so combine to single prompt
                    response_obj = client.chat_completion(
                        user_prompt=combined_prompt,
                        system_prompt="",
                        model=config.model,
                        response_schema=get_kks_chat_response_schema_with_confidence(
                            self.config.self_reported_confidence
                        ),
                    )
                else:  # ali, cst - these may need single prompt format
                    response_obj = client.chat_completion(
                        user_prompt=combined_prompt,
                        system_prompt="",
                        model=config.model,
                    )

                # Parse response
                player_assignments, explanation, confidence, _, _, _, _ = (
                    self._parse_agent_response(response_obj.text, "initial", game, agent_name)
                )

                # Add agent's response to their chat history
                response_content = json.dumps(
                    {
                        "players": [
                            {"name": name, "role": role}
                            for name, role in player_assignments.items()
                        ],
                        "confidence": confidence,
                        "explanation": explanation,
                    },
                    indent=2,
                )

                self.chat_manager.add_agent_response(
                    agent_name, response_content, "initial"
                )

                self.logger.info(f"    ✅ {agent_name} completed: {player_assignments}")
                if self.config.self_reported_confidence:
                    self.logger.info(f"    📊 {agent_name} confidence: {confidence}")
                else:
                    self.logger.info(
                        f"    📊 {agent_name} confidence: not requested (SRC disabled)"
                    )

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
                    error="",
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
                    error=str(e),
                )

        # Get proposals from all agents in parallel
        agent_names = [name for name in self.agents.keys()]
        proposals = self.parallel_processor.process_tasks(
            get_single_proposal_with_chat, agent_names, preserve_order=True
        )

        return [p for p in proposals if p is not None]

    def _validate_debate_order(self, order: List[str], game: ground_truth) -> bool:
        """Validate that the debate order is valid (each player occurs exactly once)."""
        gt_solution = self._parse_ground_truth_solution(game)
        expected_players = set(gt_solution.keys())

        # Check 1: Same length
        if len(order) != len(expected_players):
            self.logger.debug(
                f"    ❌ Invalid length: expected {len(expected_players)}, got {len(order)}"
            )
            return False

        # Check 2: All expected players are present (no missing players)
        actual_players = set(order)
        if expected_players != actual_players:
            missing_players = expected_players - actual_players
            extra_players = actual_players - expected_players
            self.logger.debug(f"    ❌ Invalid player set:")
            if missing_players:
                self.logger.debug(f"        Missing: {missing_players}")
            if extra_players:
                self.logger.debug(f"        Extra: {extra_players}")
            return False

        # Check 3: No duplicate players (each player occurs exactly once)
        if len(order) != len(set(order)):
            duplicates = [player for player in order if order.count(player) > 1]
            unique_duplicates = list(set(duplicates))
            self.logger.debug(f"    ❌ Duplicate players: {unique_duplicates}")
            return False

        # All validations passed
        self.logger.debug(f"    ✅ Valid order")
        return True

    def _select_debate_order(
        self, game: ground_truth, initial_proposals: List[AgentResponse]
    ) -> List[str]:
        """Select the debate order based on agent consensus or use ground truth order."""

        if self.config.debate_order_control == 0:
            # Use ground truth order (default behavior)
            gt_solution = self._parse_ground_truth_solution(game)
            player_names = list(gt_solution.keys())
            self.logger.info(f"🎯 Using ground truth order: {player_names}")
            return player_names

        elif self.config.debate_order_control == 1:
            # Use a dedicated GPT-5 agent to decide the order
            self.logger.info("🤖 Using GPT-5 agent to decide the debate order...")

            try:
                # Create a dedicated GPT-5 client for order selection
                from utils.utility import openai_client

                gpt5_client = openai_client(self.secret_path)

                # Create order selection prompt
                order_prompt = get_debate_order_selection_prompt(
                    game, initial_proposals
                )

                self.logger.info("  Asking GPT-5 for optimal debate order...")

                # Get response from GPT-5 using the correct method
                response_obj = gpt5_client.response_completion(
                    user_prompt=order_prompt,
                    model="gpt-5-nano",  # Use GPT-5 for reliable order selection
                    schema_format=get_debate_order_response_schema(),
                )

                if response_obj.error:
                    raise Exception(f"GPT-5 API error: {response_obj.error}")

                content = response_obj.text

                # Parse the response
                import json

                order_data = json.loads(content)
                debate_order = order_data.get("debate_order", [])
                reasoning = order_data.get("reasoning", "")

                self.logger.info(f"    GPT-5 suggests: {debate_order}")
                self.logger.info(f"    Reasoning: {reasoning}")

                # Validate the GPT-5 order
                if self._validate_debate_order(debate_order, game):
                    self.logger.info(f"✅ GPT-5 provided valid order: {debate_order}")
                    return debate_order
                else:
                    self.logger.warning(
                        "⚠️ GPT-5 provided invalid order, falling back to ground truth order"
                    )
                    gt_solution = self._parse_ground_truth_solution(game)
                    return list(gt_solution.keys())

            except Exception as e:
                self.logger.error(f"❌ GPT-5 order selection failed: {e}")
                self.logger.warning("⚠️ Falling back to ground truth order")
                gt_solution = self._parse_ground_truth_solution(game)
                return list(gt_solution.keys())

        else:
            # Invalid setting, use ground truth order
            self.logger.warning(
                f"⚠️ Invalid debate_order_control setting: {self.config.debate_order_control}, using ground truth order"
            )
            gt_solution = self._parse_ground_truth_solution(game)
            return list(gt_solution.keys())

    def _conduct_debate_for_player_with_chat(
        self,
        game: ground_truth,
        player_name: str,
        round_num: int,
        current_agent_states: List[AgentResponse],
        previous_rounds: List[DebateRound],
    ) -> List[AgentResponse]:
        """Conduct debate for a specific player's role using chat history."""

        def get_debate_response_with_chat(agent_name: str) -> AgentResponse:
            agent_info = self.agents[agent_name]
            client = agent_info["client"]
            config = agent_info["config"]

            self.logger.info(
                f"      🤖 {agent_name} debating {player_name} with chat history..."
            )

            try:
                # Create debate context for this agent's chat history
                debate_prompt = get_chat_debate_prompt_for_chat(
                    game,
                    player_name,
                    current_agent_states,
                    previous_rounds,
                    agent_name,
                    include_confidence=self.config.self_reported_confidence,
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
                        "confidence": state.confidence,
                    }
                    for state in current_agent_states
                ]

                self.chat_manager.add_other_agents_context(
                    agent_name, other_agents_data, "debate", round_num
                )

                # Get chat context for API call
                chat_context = self.chat_manager.get_chat_context_for_agent(agent_name)

                # Log the complete debate prompt with all context (ORANGE TEXT)
                self.logger.info(f"      🗣️ COMPLETE DEBATE PROMPT for {agent_name}:")
                self.logger.info(
                    f"[DEBATE PROMPT - ORANGE] \033[33m{debate_prompt}\033[0m"
                )
                self.logger.info("")

                # Log prompt length for debugging
                self.logger.info(
                    f"      📊 DEBATE PROMPT LENGTH: {len(debate_prompt)} characters"
                )
                self.logger.info("")

                # Generate dynamic debate schema for this specific player and other agents
                other_agent_names = [name for name in self.agents.keys() if name != agent_name]
                base_debate_schema = generate_kks_chat_debate_response_schema(player_name, other_agent_names)
                debate_response_schema = get_kks_chat_debate_response_schema_dynamic_with_confidence(
                    self.config.self_reported_confidence, base_schema=base_debate_schema
                )

                # Make API call
                if config.provider == "openai":
                    # All OpenAI models need combined prompt since response_completion doesn't support messages
                    combined_prompt = self._combine_chat_context_to_prompt(chat_context)
                    schema_for_openai = copy.deepcopy(debate_response_schema)
                    
                    # Check if model supports reasoning parameters
                    if config.model in ["gpt-5-nano", "gpt-5", "gpt-4o"]:
                        response_obj = client.response_completion(
                            user_prompt=combined_prompt,
                            system_prompt="",
                            model=config.model,
                            reasoning_effort=config.reasoning_effort,
                            verbosity=config.verbosity,
                            schema_format=schema_for_openai,
                        )
                    else:
                        # For models that don't support reasoning, use basic response completion
                        response_obj = client.response_completion(
                            user_prompt=combined_prompt,
                            system_prompt="",
                            model=config.model,
                            schema_format=schema_for_openai,
                        )
                elif config.provider == "gemini":
                    # Use dynamic debate-specific schema for Gemini
                    combined_prompt = self._combine_chat_context_to_prompt(chat_context)
                    
                    def strip_additional_props(schema: dict) -> dict:
                        if isinstance(schema, dict):
                            schema.pop("additionalProperties", None)
                            for v in schema.values():
                                strip_additional_props(v)
                        elif isinstance(schema, list):
                            for v in schema:
                                strip_additional_props(v)
                        return schema

                    schema_for_gemini = strip_additional_props(
                        copy.deepcopy(debate_response_schema)
                    )
                    response_obj = client.chat_completion(
                        user_prompt=combined_prompt,
                        system_prompt="",
                        model=config.model,
                        response_schema=schema_for_gemini,
                    )
                else:
                    combined_prompt = self._combine_chat_context_to_prompt(chat_context)
                    response_obj = client.chat_completion(
                        user_prompt=combined_prompt,
                        system_prompt="",
                        model=config.model,
                        response_schema=debate_response_schema,
                    )

                # Parse response
                (
                    player_assignments,
                    explanation,
                    confidence,
                    agree_with,
                    disagree_with,
                    agree_reasoning,
                    disagree_reasoning,
                ) = self._parse_agent_response(response_obj.text, "debate", None, agent_name)

                # Add agent's debate response to their chat history
                debate_response_content = json.dumps(
                    {
                        "player_role": player_name,
                        "role": player_assignments.get(player_name, "unknown"),
                        "agree_with": agree_with or [],
                        "disagree_with": disagree_with or [],
                        "agree_reasoning": agree_reasoning or "",
                        "disagree_reasoning": disagree_reasoning or "",
                        "confidence": confidence,
                    },
                    indent=2,
                )

                self.chat_manager.add_agent_response(
                    agent_name,
                    debate_response_content,
                    "debate",
                    player_focus=player_name,
                    round_number=round_num,
                )

                self.logger.info(f"      ✅ {agent_name} debate completed")
                if self.config.self_reported_confidence:
                    self.logger.info(f"      📊 {agent_name} confidence: {confidence}")

                # Log agreement/disagreement analysis
                if agree_with and len(agree_with) > 0:
                    self.logger.info(
                        f"      🤝 {agent_name} agrees with: {', '.join(agree_with)}"
                    )
                if disagree_with and len(disagree_with) > 0:
                    self.logger.info(
                        f"      ❌ {agent_name} disagrees with: {', '.join(disagree_with)}"
                    )

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
                    disagree_reasoning=disagree_reasoning,
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
                    disagree_reasoning=None,
                )

        # Get debate responses from all agents
        agent_names = [name for name in self.agents.keys()]
        responses = self.parallel_processor.process_tasks(
            get_debate_response_with_chat, agent_names, preserve_order=True
        )

        return [r for r in responses if r is not None]

    def _conduct_self_adjustment_with_chat(
        self,
        game: ground_truth,
        player_name: str,
        round_num: int,
        debate_responses: List[AgentResponse],
        previous_rounds: List[DebateRound],
        current_agent_states: List[AgentResponse],
        initial_proposals: List[AgentResponse],
    ) -> List[AgentResponse]:
        """Conduct self-adjustment phase for agents using chat history."""

        def get_adjustment_response_with_chat(agent_name: str) -> AgentResponse:
            agent_info = self.agents[agent_name]
            client = agent_info["client"]
            config = agent_info["config"]

            self.logger.info(
                f"      🔄 {agent_name} self-adjusting for {player_name} with chat history..."
            )

            try:
                # Create self-adjustment context for this agent's chat history
                adjustment_prompt = get_chat_self_adjustment_prompt_for_chat(
                    game,
                    player_name,
                    debate_responses,
                    previous_rounds,
                    agent_name,
                    config,
                    initial_proposals,
                    include_confidence=self.config.self_reported_confidence,
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
                        "confidence": response.confidence,
                    }
                    for response in debate_responses
                ]

                self.chat_manager.add_other_agents_context(
                    agent_name, debate_responses_data, "self_adjustment", round_num
                )

                # Get chat context for API call
                chat_context = self.chat_manager.get_chat_context_for_agent(agent_name)

                # Log the complete self-adjustment prompt with all context (BLUE TEXT)
                self.logger.info(
                    f"      🔄 COMPLETE SELF-ADJUSTMENT PROMPT for {agent_name}:"
                )
                self.logger.info(
                    f"[SELF-ADJUSTMENT PROMPT - BLUE] \033[34m{adjustment_prompt}\033[0m"
                )
                self.logger.info("")

                # Log prompt length for debugging
                self.logger.info(
                    f"      📊 SELF-ADJUSTMENT PROMPT LENGTH: {len(adjustment_prompt)} characters"
                )
                self.logger.info("")

                # Get player names for schema generation
                player_names = self._extract_player_names_from_game(game)
                self_adjustment_schema = (
                    get_kks_chat_self_adjustment_response_schema_with_confidence(
                        self.config.self_reported_confidence, player_names
                    )
                )
                
                # Log the schema for debugging
                self.logger.info(f"      📋 SELF-ADJUSTMENT SCHEMA: {json.dumps(self_adjustment_schema, indent=2)}")
                self.logger.info("")

                # Make API call
                if config.provider == "openai":
                    # All OpenAI models need combined prompt since response_completion doesn't support messages
                    combined_prompt = self._combine_chat_context_to_prompt(chat_context)
                    schema_for_openai = copy.deepcopy(self_adjustment_schema)

                    # Check if model supports reasoning parameters
                    if config.model in ["gpt-5-nano", "gpt-5", "gpt-4o"]:
                        response_obj = client.response_completion(
                            user_prompt=combined_prompt,
                            system_prompt="",
                            model=config.model,
                            reasoning_effort=config.reasoning_effort,
                            verbosity=config.verbosity,
                            schema_format=schema_for_openai,
                        )
                    else:
                        # For models that don't support reasoning, use basic response completion
                        response_obj = client.response_completion(
                            user_prompt=combined_prompt,
                            system_prompt="",
                            model=config.model,
                            schema_format=schema_for_openai,
                        )
                elif config.provider == "gemini":
                    combined_prompt = self._combine_chat_context_to_prompt(chat_context)

                    def strip_additional_props(schema: dict) -> dict:
                        if isinstance(schema, dict):
                            schema.pop("additionalProperties", None)
                            for v in schema.values():
                                strip_additional_props(v)
                        elif isinstance(schema, list):
                            for v in schema:
                                strip_additional_props(v)
                        return schema

                    schema_for_gemini = strip_additional_props(
                        copy.deepcopy(self_adjustment_schema)
                    )

                    response_obj = client.chat_completion(
                        user_prompt=combined_prompt,
                        system_prompt="",
                        model=config.model,
                        response_schema=schema_for_gemini,
                    )
                else:
                    combined_prompt = self._combine_chat_context_to_prompt(chat_context)

                    response_obj = client.chat_completion(
                        user_prompt=combined_prompt,
                        system_prompt="",
                        model=config.model,
                        response_schema=self_adjustment_schema,
                    )

                # Parse response
                player_assignments, explanation, confidence, _, _, _, _ = (
                    self._parse_agent_response(response_obj.text, "self_adjustment", game, agent_name)
                )

                # Add agent's self-adjustment response to their chat history
                adjustment_response_content = json.dumps(
                    {
                        "players": [
                            {"name": name, "role": role}
                            for name, role in player_assignments.items()
                        ],
                        "confidence": confidence,
                        "explanation": explanation,
                    },
                    indent=2,
                )

                self.chat_manager.add_agent_response(
                    agent_name,
                    adjustment_response_content,
                    "self_adjustment",
                    player_focus=player_name,
                    round_number=round_num,
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
                    error="",
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
                    error=str(e),
                )

        # Get adjustment responses from all agents
        agent_names = [name for name in self.agents.keys()]
        responses = self.parallel_processor.process_tasks(
            get_adjustment_response_with_chat, agent_names, preserve_order=True
        )

        return [r for r in responses if r is not None]

    def _conduct_final_discussion_and_vote_with_chat(
        self,
        game: ground_truth,
        initial_proposals: List[AgentResponse],
        debate_rounds: List[DebateRound],
    ) -> Dict[str, str]:
        """Conduct final discussion where agents reconsider everything using chat history."""
        self.logger.info(
            "Conducting final discussion and fresh voting with chat history..."
        )

        # Get all player names
        player_names = list(initial_proposals[0].player_role_assignments.keys())

        # Create final discussion context for each agent's chat history
        final_discussion_prompt = get_chat_final_discussion_prompt_for_chat(
            game,
            initial_proposals,
            debate_rounds,
            include_confidence=self.config.self_reported_confidence,
        )

        # Get fresh votes from all agents
        fresh_votes = []
        agent_names = [name for name in self.agents.keys()]

        for agent_name in agent_names:
            agent_info = self.agents[agent_name]
            client = agent_info["client"]
            config = agent_info["config"]

            self.logger.info(
                f"  🤖 {agent_name} making final decision with chat history..."
            )

            try:
                # Add final discussion context to this agent's chat history
                self.chat_manager.create_final_discussion_message(
                    agent_name, final_discussion_prompt
                )

                # Add all previous responses as context
                all_responses_data = []

                # Add initial proposals
                for proposal in initial_proposals:
                    all_responses_data.append(
                        {
                            "agent_name": proposal.agent_name,
                            "player_role_assignments": proposal.player_role_assignments,
                            "explanation": proposal.explanation,
                            "confidence": proposal.confidence,
                            "phase": "initial",
                        }
                    )

                # Add debate round responses
                for round_data in debate_rounds:
                    for response in round_data.agent_responses:
                        all_responses_data.append(
                            {
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
                                "player_focus": round_data.player_name,
                            }
                        )

                self.chat_manager.add_other_agents_context(
                    agent_name, all_responses_data, "final"
                )

                # Get chat context for API call
                chat_context = self.chat_manager.get_chat_context_for_agent(agent_name)

                # Log the complete final discussion prompt with all context (GREEN TEXT)
                self.logger.info(
                    f"  🎯 COMPLETE FINAL DISCUSSION PROMPT for {agent_name}:"
                )
                self.logger.info(
                    f"[FINAL DISCUSSION PROMPT - GREEN] \033[32m{final_discussion_prompt}\033[0m"
                )
                self.logger.info("")

                # Log prompt length for debugging
                self.logger.info(
                    f"  📊 FINAL DISCUSSION PROMPT LENGTH: {len(final_discussion_prompt)} characters"
                )
                self.logger.info("")

                # Make API call for final decision
                if config.provider == "openai":
                    if config.model in ["gpt-5-nano", "gpt-5", "gpt-4o"]:
                        # All OpenAI models need combined prompt since response_completion doesn't support messages
                        combined_prompt = self._combine_chat_context_to_prompt(
                            chat_context
                        )
                        response_obj = client.response_completion(
                            user_prompt=combined_prompt,
                            system_prompt="",
                            model=config.model,
                            reasoning_effort=config.reasoning_effort,
                            verbosity=config.verbosity,
                        )
                    else:
                        # For models that don't support reasoning, combine to single prompt
                        combined_prompt = self._combine_chat_context_to_prompt(
                            chat_context
                        )
                        response_obj = client.response_completion(
                            user_prompt=combined_prompt,
                            system_prompt="",
                            model=config.model,
                        )
                elif config.provider == "gemini":
                    combined_prompt = self._combine_chat_context_to_prompt(chat_context)
                    response_obj = client.chat_completion(
                        user_prompt=combined_prompt,
                        system_prompt="",
                        model=config.model,
                        response_schema=get_kks_chat_response_schema_with_confidence(
                            self.config.self_reported_confidence
                        ),
                    )
                else:
                    combined_prompt = self._combine_chat_context_to_prompt(chat_context)
                    response_obj = client.chat_completion(
                        user_prompt=combined_prompt,
                        system_prompt="",
                        model=config.model,
                    )

                # Parse response
                player_assignments, explanation, confidence, _, _, _, _ = (
                    self._parse_agent_response(response_obj.text, "final", game, agent_name)
                )

                # Add agent's final response to their chat history
                final_response_content = json.dumps(
                    {
                        "players": [
                            {"name": name, "role": role}
                            for name, role in player_assignments.items()
                        ],
                        "confidence": confidence,
                        "explanation": explanation,
                    },
                    indent=2,
                )

                self.chat_manager.add_agent_response(
                    agent_name, final_response_content, "final"
                )

                fresh_votes.append(
                    {
                        "agent_name": agent_name,
                        "player_assignments": player_assignments,
                        "explanation": explanation,
                        "confidence": confidence,
                    }
                )

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
                fresh_votes.append(
                    {
                        "agent_name": agent_name,
                        "player_assignments": last_assignment,
                        "explanation": f"Error: {str(e)}",
                        "confidence": 1.0,  # Low confidence for errors
                    }
                )

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
                self.logger.info(
                    f"  {player_name}: {majority_role} (fresh votes: {dict(vote_counts)})"
                )

        return final_vote

    def _combine_chat_context_to_prompt(
        self, chat_context: List[Dict[str, str]]
    ) -> str:
        """Combine chat context into a single prompt for models that don't support chat format."""
        prompt_parts = []

        for msg in chat_context:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                prompt_parts.append(f"SYSTEM: {content}")
            elif role == "user":
                prompt_parts.append(f"USER: {content}")
            elif role == "assistant":
                prompt_parts.append(f"ASSISTANT: {content}")

        return "\n\n".join(prompt_parts)

    def _get_supervisor_decision(
        self,
        game: ground_truth,
        initial_proposals: List[AgentResponse],
        debate_rounds: List[DebateRound],
    ) -> Dict[str, str]:
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
            supervisor_prompt = get_chat_supervisor_prompt_for_chat(
                game, initial_proposals, debate_rounds
            )

            # Make API call
            system_prompt = get_kks_chat_system_prompt_with_confidence(
                game.num_player, self.config.self_reported_confidence
            )
            response_schema = get_kks_chat_response_schema_with_confidence(
                self.config.self_reported_confidence
            )

            if self.config.supervisor_provider == "openai":
                # Check if supervisor model supports reasoning parameters
                if self.config.supervisor_model in ["gpt-5-nano", "gpt-5", "gpt-4o"]:
                    response_obj = supervisor_client.response_completion(
                        user_prompt=supervisor_prompt,
                        system_prompt=system_prompt,
                        model=self.config.supervisor_model,
                        reasoning_effort="high",
                        verbosity="high",
                    )
                else:
                    # Use regular response completion for models that don't support reasoning
                    response_obj = supervisor_client.response_completion(
                        user_prompt=supervisor_prompt,
                        system_prompt=system_prompt,
                        model=self.config.supervisor_model,
                    )
            elif self.config.supervisor_provider == "gemini":
                response_obj = supervisor_client.chat_completion(
                    user_prompt=supervisor_prompt,
                    system_prompt=system_prompt,
                    model=self.config.supervisor_model,
                    response_schema=response_schema,
                )
            else:
                response_obj = supervisor_client.chat_completion(
                    user_prompt=supervisor_prompt,
                    system_prompt=system_prompt,
                    model=self.config.supervisor_model,
                )

            # Parse supervisor response
            player_assignments, explanation, confidence, _, _, _, _ = (
                self._parse_agent_response(response_obj.text, "final", game, "Supervisor")
            )
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
            filename = (
                f"chat_history_{agent_name}_game{session.game_id}_{timestamp}.json"
            )
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
                        "round_number": msg.round_number,
                    }
                    for msg in history.messages
                ],
            }

            with open(filepath, "w", encoding="utf-8") as f:
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

            self.agents[agent_name] = {"config": agent_config, "client": client}

    def _setup_logging(self, game_id: int = 1, create_log_file: bool = True):
        """Setup logging for the debate system (borrowed from traditional system)."""
        import logging
        from datetime import datetime

        # Only setup logging once per game to avoid multiple log files
        if (
            hasattr(self, "logger")
            and self.logger is not None
            and hasattr(self, "current_game_id")
            and self.current_game_id == game_id
        ):
            return

        # Use a more unique timestamp with microseconds to avoid conflicts in parallel processing
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[
            :-3
        ]  # Include milliseconds
        log_filename = f"debate_log_game{game_id}_{timestamp}.log"
        html_filename = f"debate_log_game{game_id}_{timestamp}.html"
        log_path = os.path.join(self.organized_output_path, log_filename)
        html_path = os.path.join(self.organized_output_path, html_filename)

        # Create logger with game-specific name
        logger_name = f"chat_debate_system_game{game_id}_{timestamp}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)

        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Only create file handler if requested
        if create_log_file:
            # Store the log paths for later use
            self.log_file_path = log_path
            self.html_file_path = html_path
            # Don't create the file handler yet - we'll create it when we have content to log

        # Always log to console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # Track current game ID to avoid duplicate setup
        self.current_game_id = game_id

    def _ensure_log_file_handler(self):
        """Ensure the log file handler is created (borrowed from traditional system)."""
        import logging

        # Check if we already have a file handler
        has_file_handler = any(
            isinstance(handler, logging.FileHandler) for handler in self.logger.handlers
        )

        if not has_file_handler and hasattr(self, "log_file_path"):
            # Create file handler
            file_handler = logging.FileHandler(self.log_file_path, encoding="utf-8")
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            # Create HTML file handler
            if hasattr(self, "html_file_path"):
                html_handler = HTMLFileHandler(self.html_file_path)
                html_handler.setLevel(logging.INFO)
                formatter = logging.Formatter(
                    "%(asctime)s - %(levelname)s - %(message)s"
                )
                html_handler.setFormatter(formatter)
                self.logger.addHandler(html_handler)

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
            pattern = r"(\w+)\s+is\s+a\s+(knight|knave|spy)\."
            matches = re.findall(pattern, solution_text, re.IGNORECASE)

            for player_name, role in matches:
                pairs.append((player_name, role.lower()))

            return dict(pairs)

        except Exception as e:
            self.logger.error(f"Error parsing ground truth solution: {e}")
            self.logger.error(f"Solution text: {game.text_solution}")
            return {}

    def _extract_player_names_from_game(self, game: ground_truth) -> List[str]:
        """Extract all player names from the game's text solution."""
        import re
        solution_text = game.text_solution.strip()
        pattern = r'(\w+)\s+is\s+a\s+(knight|knave|spy)\.'
        matches = re.findall(pattern, solution_text, re.IGNORECASE)
        player_names = [match[0] for match in matches]
        return player_names

    def _fill_missing_players_with_fallback(
        self, 
        player_assignments: Dict[str, str], 
        all_player_names: List[str], 
        agent_name: str, 
        phase: str
    ) -> Dict[str, str]:
        """
        Fill missing player assignments using the agent's latest available assignments.
        If an agent missed assigning roles to several players, assign those missing players 
        the roles same as the latest available assignments from the same agent.
        """
        if not all_player_names:
            return player_assignments
            
        missing_players = [name for name in all_player_names if name not in player_assignments]
        
        if not missing_players:
            # No missing players, return as is
            return player_assignments
            
        self.logger.warning(
            f"Agent {agent_name} in {phase} phase missing assignments for players: {missing_players}"
        )
        
        # Get available role assignments (excluding special keys like "role", "error")
        available_assignments = {
            name: role for name, role in player_assignments.items() 
            if name in all_player_names and role in ['knight', 'knave', 'spy']
        }
        
        if not available_assignments:
            # No valid assignments available, assign default role (knight) to missing players
            self.logger.warning(
                f"No valid assignments available for {agent_name}, using default role 'knight' for missing players"
            )
            for missing_player in missing_players:
                player_assignments[missing_player] = 'knight'
        else:
            # Use the most recent assignment as fallback
            # Since dict maintains insertion order in Python 3.7+, the last item is the most recent
            fallback_role = list(available_assignments.values())[-1]
            
            self.logger.info(
                f"Using fallback role '{fallback_role}' for missing players {missing_players} from agent {agent_name}"
            )
            
            for missing_player in missing_players:
                player_assignments[missing_player] = fallback_role
                
        return player_assignments

    def _parse_agent_response(
        self, response_text: str, phase: str = "initial", game: ground_truth = None, agent_name: str = None
    ) -> Tuple[
        Dict[str, str],
        str,
        float,
        Optional[List[str]],
        Optional[List[str]],
        Optional[str],
        Optional[str],
    ]:
        """Parse agent response to extract player assignments, explanation, confidence, and debate analysis."""
        try:
            # First try to extract JSON from the response (handle extra text after JSON)
            json_text = self._extract_json_from_response(response_text)
            self.logger.debug(f"Extracted JSON text: {json_text[:200]}...")

            response_data = json.loads(json_text)
            self.logger.debug(f"Parsed JSON data: {response_data}")

            # Extract player assignments (handle different formats based on phase)
            player_assignments = {}
            if phase == "debate":
                # Debate format: {"player_role": "PlayerName", "role": "knight/knave/spy"}
                if "player_role" in response_data and "role" in response_data:
                    player_name = response_data.get("player_role", "")
                    role = response_data.get("role", "")
                    if player_name and role:
                        player_assignments[player_name] = role
                        # Also store as "role" key for consistent access
                        player_assignments["role"] = role
                        self.logger.debug(
                            f"Added debate assignment: {player_name} -> {role}"
                        )
                else:
                    self.logger.warning(
                        f"Missing player_role or role in debate response: {response_data}"
                    )
            elif "players" in response_data and isinstance(
                response_data["players"], list
            ):
                # New format with players array
                self.logger.debug(
                    f"Using players array format with {len(response_data['players'])} players"
                )
                for player_info in response_data["players"]:
                    if (
                        isinstance(player_info, dict)
                        and "name" in player_info
                        and "role" in player_info
                    ):
                        name = player_info.get("name", "")
                        role = player_info.get("role", "")
                        if name and role:
                            player_assignments[name] = role
                            self.logger.debug(
                                f"Added player assignment: {name} -> {role}"
                            )
                self.logger.debug(f"Final player_assignments: {player_assignments}")
            else:
                # Old format - direct name: role mapping
                self.logger.debug("Using direct mapping format")
                for key, value in response_data.items():
                    if key not in [
                        "explanation",
                        "confidence",
                        "agree_with",
                        "disagree_with",
                        "agree_reasoning",
                        "disagree_reasoning",
                        "player_role",
                        "role",
                    ] and isinstance(value, str):
                        player_assignments[key] = value
                        self.logger.debug(f"Added direct assignment: {key} -> {value}")
                self.logger.debug(f"Final player_assignments: {player_assignments}")

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

            # Apply fallback logic for missing players (only for phases that need complete assignments)
            if phase in ["initial", "self_adjustment", "final"] and game is not None and agent_name is not None:
                all_player_names = self._extract_player_names_from_game(game)
                player_assignments = self._fill_missing_players_with_fallback(
                    player_assignments, all_player_names, agent_name, phase
                )

            return (
                player_assignments,
                explanation,
                confidence,
                agree_with,
                disagree_with,
                agree_reasoning,
                disagree_reasoning,
            )

        except Exception as e:
            self.logger.error(f"Error parsing agent response: {e}")
            self.logger.error(f"Raw response: {response_text}")
            self.logger.error(f"Response length: {len(response_text)} characters")
            # Return default values instead of empty dict to avoid {} in logs
            return (
                {"error": "parsing_failed"},
                f"Error parsing response: {str(e)}",
                5.0,
                None,
                None,
                None,
                None,
            )

    def _extract_players_from_malformed_json(self, json_text: str) -> List[Dict[str, str]]:
        """Extract player data from malformed JSON like the Qwen response."""
        import re
        
        players_data = []
        
        # Look for player name-role pairs in the malformed JSON
        # Pattern: {"name": "Bob", "role": "knave"}, or variations with missing commas/braces
        name_role_pattern = r'"name"\s*:\s*"([^"]+)"\s*,?\s*"role"\s*:\s*"([^"]+)"'
        
        matches = re.findall(name_role_pattern, json_text)
        for name, role in matches:
            if name and role and role in ['knight', 'knave', 'spy']:
                players_data.append({"name": name, "role": role})
        
        # Also look for the specific malformed pattern in the Qwen response
        # {"name": "Frank", "role": "knight",\n        "name": "Grace",\n        "role": "knight",
        broken_pattern = r'"name"\s*:\s*"([^"]+)"\s*,?\s*"role"\s*:\s*"([^"]+)"\s*,?\s*"name"'
        broken_matches = re.findall(broken_pattern, json_text)
        for name, role in broken_matches:
            if name and role and role in ['knight', 'knave', 'spy']:
                # Check if we already have this player
                if not any(p['name'] == name for p in players_data):
                    players_data.append({"name": name, "role": role})
        
        # Remove duplicates while preserving order
        seen_names = set()
        unique_players = []
        for player in players_data:
            if player['name'] not in seen_names:
                unique_players.append(player)
                seen_names.add(player['name'])
        
        return unique_players

    def _extract_json_from_response(self, response_text: str) -> str:
        """Extract JSON from response text, handling extra text after JSON and double braces from GPT-5 models."""
        import json
        import re
        
        # First, try to handle double braces that GPT-5 models sometimes return
        # Replace double braces with single braces
        normalized_text = response_text.replace("{{", "{").replace("}}", "}")

        # Log if we had to normalize double braces for debugging
        if "{{" in response_text or "}}" in response_text:
            self.logger.debug(
                f"Normalized double braces in response: {response_text[:100]}..."
            )

        # Strategy 1: Try to find a complete, valid JSON object first
        try:
            # Look for JSON object boundaries with proper string handling
            start_idx = normalized_text.find("{")
            if start_idx == -1:
                raise ValueError("No JSON object found")

            # Use a more robust approach to find the end of JSON
            json_candidate = self._find_complete_json_object(normalized_text[start_idx:])
            if json_candidate:
                # Validate that it's proper JSON
                json.loads(json_candidate)
                return json_candidate
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.debug(f"Initial JSON extraction failed: {e}")

        # Strategy 2: Try to repair malformed JSON
        try:
            json_candidate = normalized_text[start_idx:] if start_idx != -1 else normalized_text
            repaired_json = self._attempt_json_repair(json_candidate)
            if repaired_json:
                self.logger.warning(f"Repaired malformed JSON from response")
                return repaired_json
        except Exception as e:
            self.logger.debug(f"JSON repair failed: {e}")

        # Strategy 3: Extract key-value pairs using regex as last resort
        try:
            extracted_json = self._extract_json_with_regex(normalized_text)
            if extracted_json:
                self.logger.warning(f"Extracted JSON using regex fallback")
                return extracted_json
        except Exception as e:
            self.logger.debug(f"Regex extraction failed: {e}")

        raise ValueError("Could not extract valid JSON from response")

    def _find_complete_json_object(self, text: str) -> str:
        """Find a complete JSON object, handling strings properly."""
        import json
        
        brace_count = 0
        in_string = False
        escape_next = False
        
        for i, char in enumerate(text):
            if escape_next:
                escape_next = False
                continue
                
            if char == '\\' and in_string:
                escape_next = True
                continue
                
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
                
            if not in_string:
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        candidate = text[:i + 1]
                        try:
                            json.loads(candidate)
                            return candidate
                        except json.JSONDecodeError:
                            continue
        
        return None

    def _extract_json_with_regex(self, text: str) -> str:
        """Extract JSON structure using regex patterns as a fallback."""
        import json
        import re
        
        # Try to extract key information using regex patterns
        extracted_data = {}
        
        # Look for player_role and role (debate format)
        player_role_match = re.search(r'"player_role"\s*:\s*"([^"]+)"', text)
        role_match = re.search(r'"role"\s*:\s*"([^"]+)"', text)
        
        if player_role_match and role_match:
            extracted_data["player_role"] = player_role_match.group(1)
            extracted_data["role"] = role_match.group(1)
        
        # Look for agree_with array
        agree_with_match = re.search(r'"agree_with"\s*:\s*\[([^\]]*)\]', text)
        if agree_with_match:
            agree_content = agree_with_match.group(1)
            # Extract quoted strings from the array
            agree_items = re.findall(r'"([^"]+)"', agree_content)
            extracted_data["agree_with"] = agree_items
        else:
            extracted_data["agree_with"] = []
        
        # Look for disagree_with array
        disagree_with_match = re.search(r'"disagree_with"\s*:\s*\[([^\]]*)\]', text)
        if disagree_with_match:
            disagree_content = disagree_with_match.group(1)
            disagree_items = re.findall(r'"([^"]+)"', disagree_content)
            extracted_data["disagree_with"] = disagree_items
        else:
            extracted_data["disagree_with"] = []
        
        # Look for reasoning fields - handle unterminated strings
        agree_reasoning_match = re.search(r'"agree_reasoning"\s*:\s*"([^"]*)', text)
        if agree_reasoning_match:
            extracted_data["agree_reasoning"] = agree_reasoning_match.group(1)
        else:
            extracted_data["agree_reasoning"] = ""
        
        disagree_reasoning_match = re.search(r'"disagree_reasoning"\s*:\s*"([^"]*)', text)
        if disagree_reasoning_match:
            extracted_data["disagree_reasoning"] = disagree_reasoning_match.group(1)
        else:
            extracted_data["disagree_reasoning"] = ""
        
        # Look for explanation field
        explanation_match = re.search(r'"explanation"\s*:\s*"([^"]*)', text)
        if explanation_match:
            extracted_data["explanation"] = explanation_match.group(1)
        
        # Look for players array format
        players_match = re.search(r'"players"\s*:\s*\[([^\]]*)\]', text, re.DOTALL)
        if players_match:
            players_content = players_match.group(1)
            players = []
            # Extract individual player objects
            player_matches = re.findall(r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"role"\s*:\s*"([^"]+)"\s*\}', players_content)
            for name, role in player_matches:
                players.append({"name": name, "role": role})
            if players:
                extracted_data["players"] = players
        
        # Only return if we extracted meaningful data
        if extracted_data:
            return json.dumps(extracted_data)
        
        return None

    def _fix_unterminated_strings(self, text: str) -> str:
        """Fix unterminated strings and extra text in JSON responses."""
        import re
        import json
        
        # Strategy 1: Find the main JSON structure and terminate at logical boundaries
        # Look for key patterns that indicate where the JSON should end
        
        # Find the start of JSON
        start_idx = text.find("{")
        if start_idx == -1:
            return None
            
        # Look for patterns that indicate the end of meaningful JSON content
        # Common patterns: closing of disagree_reasoning field, end of explanation, etc.
        
        # Try to find a reasonable cutoff point
        cutoff_patterns = [
            # End of disagree_reasoning field followed by closing brace
            r'"disagree_reasoning"\s*:\s*"[^"]*"\s*}',
            # End of agree_reasoning field followed by closing brace  
            r'"agree_reasoning"\s*:\s*"[^"]*"\s*}',
            # End of explanation field followed by closing brace
            r'"explanation"\s*:\s*"[^"]*"\s*}',
            # End of role field in debate format
            r'"role"\s*:\s*"[^"]*"\s*}',
        ]
        
        json_candidate = text[start_idx:]
        
        for pattern in cutoff_patterns:
            match = re.search(pattern, json_candidate)
            if match:
                # Try cutting at this point
                candidate = json_candidate[:match.end()]
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    continue
        
        # Strategy 2: Fix specific unterminated string issues
        # Look for unterminated strings and try to close them properly
        try:
            # Find unterminated strings by looking for quotes that aren't properly closed
            fixed_text = json_candidate
            
            # Handle the specific case from the error log where there's extra text after JSON
            # Pattern: "...reasoning":"Both agents conclude Kate is a knight.","disagree_reasoning":"} }  (Note: The trailing characters..."
            
            # Look for the pattern where reasoning field has extra text
            reasoning_pattern = r'("(?:agree_|disagree_)?reasoning"\s*:\s*"[^"]*")\s*,?\s*"[^"]*"\s*}\s*}\s*\([^)]*\)[^}]*'
            match = re.search(reasoning_pattern, fixed_text)
            if match:
                # Keep only the valid reasoning part and close the JSON
                valid_part = match.group(1)
                # Find the position and reconstruct
                before_reasoning = fixed_text[:match.start()]
                # Close the JSON properly
                fixed_text = before_reasoning + valid_part + "}"
                try:
                    json.loads(fixed_text)
                    return fixed_text
                except json.JSONDecodeError:
                    pass
            
            # Strategy 3: Handle cases where there's a valid JSON followed by garbage
            # Try to find the longest valid JSON prefix
            for i in range(len(fixed_text) - 1, 0, -1):
                if fixed_text[i] == '}':
                    candidate = fixed_text[:i + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            pass
            
        return None

    def _attempt_json_repair(self, json_text: str) -> str:
        """Attempt to repair common JSON formatting issues."""
        try:
            # Common repair strategies for malformed JSON
            repaired = json_text.strip()
            import re
            import json

            # Strategy 0: Handle unterminated strings and extra text
            # This addresses the specific error: "Unterminated string starting at: line 1 column 209"
            try:
                # Find the first complete JSON-like structure and terminate unterminated strings
                fixed_json = self._fix_unterminated_strings(repaired)
                if fixed_json:
                    json.loads(fixed_json)  # Validate
                    self.logger.info("Successfully repaired JSON with unterminated strings")
                    return fixed_json
            except Exception as e:
                self.logger.debug(f"Unterminated string repair failed: {e}")

            # Strategy 1: Handle the specific Qwen malformed JSON case
            # Look for malformed players array with missing commas and broken explanation
            if '"players"' in repaired and '"name":' in repaired:
                try:
                    # Extract players array data even if malformed
                    players_data = self._extract_players_from_malformed_json(repaired)
                    if players_data:
                        # Extract any explanation fragments
                        explanation_fragments = re.findall(r'"([^"]*(?:manager|single-spy|constraint|solution|consistent)[^"]*)"', repaired)
                        explanation = " ".join(explanation_fragments) if explanation_fragments else "Extracted from malformed JSON response."
                        
                        # Construct a valid JSON
                        valid_json = {
                            "players": players_data,
                            "explanation": explanation
                        }
                        
                        candidate = json.dumps(valid_json)
                        json.loads(candidate)  # Validate
                        self.logger.info("Successfully repaired malformed Qwen JSON response")
                        return candidate
                except Exception as e:
                    self.logger.debug(f"Strategy 1 failed: {e}")

            # Strategy 2: Try to find where the JSON should end and add missing braces
            # Look for patterns like "explanation": "..." that indicate the end of content
            explanation_match = re.search(
                r'"explanation"\s*:\s*"([^"]*)"', repaired, re.DOTALL
            )
            if explanation_match:
                # Find the end of the explanation field
                explanation_end = explanation_match.end()

                # Count braces up to this point
                text_up_to_explanation = repaired[:explanation_end]
                open_braces = text_up_to_explanation.count("{")
                close_braces = text_up_to_explanation.count("}")

                # Add missing closing braces
                missing_braces = open_braces - close_braces
                if missing_braces > 0:
                    # Try adding the missing braces
                    candidate = text_up_to_explanation + "}" * missing_braces

                    # Validate the repair attempt
                    try:
                        json.loads(candidate)
                        self.logger.info(
                            f"Successfully repaired JSON by adding {missing_braces} closing braces"
                        )
                        return candidate
                    except json.JSONDecodeError:
                        pass

            # Strategy 2: Try to fix incomplete player objects
            # Look for patterns like '"name": "PlayerName",\n"role": "role"\n],\n' (missing closing brace)
            player_pattern = (
                r'"name"\s*:\s*"([^"]+)"\s*,\s*"role"\s*:\s*"([^"]+)"\s*(?=\s*[,\]])'
            )

            # Find all player objects and check if they're properly closed
            matches = list(re.finditer(player_pattern, repaired))
            if matches:
                # Work backwards through matches to avoid index shifting
                for match in reversed(matches):
                    # Check if this player object is missing a closing brace
                    before_match = repaired[: match.start()]
                    after_match = repaired[match.end() :]

                    # Count braces in the player object context
                    recent_text = repaired[
                        max(0, match.start() - 50) : match.end() + 10
                    ]

                    # If we find a pattern like '"role": "knight"\n],' it's missing a closing brace
                    if re.search(r'"role"\s*:\s*"[^"]+"\s*(?=\s*[,\]])', recent_text):
                        # Insert closing brace after the role
                        insertion_point = match.end()
                        repaired = (
                            repaired[:insertion_point]
                            + "}"
                            + repaired[insertion_point:]
                        )

                        # Test if this fixes the JSON
                        try:
                            import json

                            json.loads(repaired)
                            self.logger.info(
                                "Successfully repaired JSON by adding missing player object closing brace"
                            )
                            return repaired
                        except json.JSONDecodeError:
                            # Revert the change and try next match
                            repaired = (
                                repaired[:insertion_point]
                                + repaired[insertion_point + 1 :]
                            )

            return None  # Could not repair

        except Exception as e:
            self.logger.debug(f"JSON repair attempt failed: {e}")
            return None

    def _create_debate_summary(
        self, responses: List[AgentResponse], player_name: str
    ) -> str:
        """Create a summary of debate responses for a specific player."""
        if not responses:
            return "No responses received."

        summary_parts = []
        for response in responses:
            # Handle different response formats based on phase
            if response.phase == "debate":
                # For debate phase, the structure is {"player_role": "PlayerName", "role": "knight/knave/spy"}
                role = response.player_role_assignments.get("role", "unknown")
            else:
                # For other phases, use the player name as key
                role = response.player_role_assignments.get(player_name, "unknown")
            summary_parts.append(f"{response.agent_name}: {role}")

        return "; ".join(summary_parts)

    def _check_consensus(
        self, responses: List[AgentResponse], player_name: str
    ) -> Optional[str]:
        """Check if there's consensus among agents for a specific player's role."""
        if not responses:
            return None

        # Count votes for each role
        role_counts = {}
        for response in responses:
            # Handle different response formats based on phase
            if response.phase == "debate":
                # For debate phase, the structure is {"player_role": "PlayerName", "role": "knight/knave/spy"}
                role = response.player_role_assignments.get("role", "unknown")
            else:
                # For other phases, use the player name as key
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

    def _create_performance_tracking(
        self,
        game: ground_truth,
        initial_proposals: List[AgentResponse],
        debate_rounds: List[DebateRound],
        final_vote: Dict[str, str],
        supervisor_decision: Optional[Dict[str, str]],
    ) -> Dict[str, Any]:
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
            "supervisor_used": supervisor_decision is not None,
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

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(session_dict, f, indent=2, ensure_ascii=False)

        self.logger.info(f"💾 Debate session saved to: {filepath}")

    def _generate_visualizations(self, session: DebateSession):
        """Generate visualizations for the debate session."""
        # This is a placeholder - you can implement visualization generation here
        self.logger.info(
            "📊 Visualization generation placeholder - implement as needed"
        )

    def run_batch_debate(self, games: List[ground_truth]) -> List[DebateSession]:
        """Run debates for multiple games sequentially."""
        sessions = []

        for i, game in enumerate(games, 1):
            self.logger.info(
                f"\n🎮 Starting Game {i}/{len(games)}: Game {game.game_id}"
            )
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
                    performance_tracking={"error": str(e)},
                )
                sessions.append(failed_session)

        return sessions

    def run_parallel_batch_debate(
        self, games: List[ground_truth]
    ) -> List[DebateSession]:
        """Run debates for multiple games in parallel."""

        # Use parallel processor to run all games (borrowed from traditional system)
        print(f"🚀 Starting parallel processing of {len(games)} games")
        print(
            f"🔧 Using {self.parallel_processor.num_workers} workers for game-level parallelism"
        )

        # Create a wrapper function that handles logging setup for each game
        def run_game_with_logging(game: ground_truth) -> DebateSession:
            # Create a temporary system instance for this specific game without logging
            temp_system = ChatHistoryDebateSystem(
                self.config, self.secret_path, setup_logging=False
            )
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

        sessions = self.parallel_processor.process_tasks(
            run_game_with_logging, games, preserve_order=True
        )

        # Filter out None results
        sessions = [s for s in sessions if s is not None]

        print(f"🏁 Completed parallel batch debate: {len(sessions)} sessions")
        return sessions
