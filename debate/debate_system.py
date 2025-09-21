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
from utils.utility import openai_client, gemini_client, ali_client, cstcloud, ParallelProcessor
from prompts import (
    kks_system_prompt,
    kks_response_schema,
    get_kks_system_prompt_with_confidence,
    get_kks_response_schema_with_confidence,
    get_kks_debate_response_schema_with_confidence
)

# Import visualizer (with try-except to handle potential import issues)
try:
    from debate.debate_visualizer import DebateVisualizer
    VISUALIZER_AVAILABLE = True
except ImportError as e:
    VISUALIZER_AVAILABLE = False
    print(f"⚠️ Visualizer not available: {e}")

@dataclass
class AgentResponse:
    """Response from a single agent."""
    agent_name: str
    game_id: int
    round_number: int
    phase: str  # "initial", "debate", "self_adjustment", "final"
    player_role_assignments: Dict[str, str]  # {player_name: role}
    explanation: str
    confidence: float = 5.0
    response_obj: Optional[response_format] = None
    timestamp: str = ""
    error: str = ""
    # New fields for debate phase
    agree_with: Optional[List[str]] = None  # List of agent names this agent agrees with
    disagree_with: Optional[List[str]] = None  # List of agent names this agent disagrees with
    agree_reasoning: Optional[str] = None  # Reasoning for agreements
    disagree_reasoning: Optional[str] = None  # Reasoning for disagreements

@dataclass
class DebateRound:
    """A single round of debate for a specific player's role."""
    player_name: str
    round_number: int
    agent_responses: List[AgentResponse]  # This will contain both debate and self_adjustment phases
    debate_summary: str = ""
    consensus_reached: bool = False
    majority_role: Optional[str] = None

@dataclass
class DebateSession:
    """Complete debate session for a single game."""
    game_id: int
    game_text: str
    ground_truth_solution: Dict[str, str]
    initial_proposals: List[AgentResponse]
    debate_rounds: List[DebateRound]
    final_vote: Optional[Dict[str, str]] = None
    supervisor_decision: Optional[Dict[str, str]] = None
    performance_tracking: Dict[str, Any] = None

class MultiAgentDebateSystem:
    """Multi-agent debate system for knight-knaves-spy games.
    
    This system supports two levels of parallel processing:
    1. Agent-level parallelism: Multiple agents debate simultaneously within each game
    2. Game-level parallelism: Multiple games can be processed simultaneously
    
    The system maintains separate logging for each game case, ensuring clean separation
    of results and debugging information.
    """
    
    def __init__(self, debate_config: DebateConfig, secret_path: str = "secret.json", setup_logging: bool = True):
        self.config = debate_config
        self.secret_path = secret_path
        self.agents = {}
        # Parallel processor for agent-level parallelism (debate rounds)
        self.parallel_processor = ParallelProcessor(num_workers=len(debate_config.agents))
        # Parallel processor for game-level parallelism (multiple games)
        self.game_parallel_processor = ParallelProcessor(num_workers=debate_config.game_parallel_workers)
        
        # Store config for later use
        self.debate_config = debate_config
        
        # Initialize a basic logger first
        import logging
        self.logger = logging.getLogger("debate_system")
        if not self.logger.handlers:
            # Add a console handler if no handlers exist
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            self.logger.setLevel(logging.INFO)
        
        # Setup detailed logging (only if requested)
        if setup_logging:
            self._setup_logging()
        
        # Initialize agent clients
        self._initialize_agents()
    
    def _setup_logging(self, game_id: int = 1, create_log_file: bool = True):
        """Setup logging to capture all printed output."""
        # Create organized output directory for this specific game
        self.organized_output_path = self.debate_config.get_organized_output_path(game_id)
        os.makedirs(self.organized_output_path, exist_ok=True)
        
        # Use a more unique timestamp with microseconds to avoid conflicts in parallel processing
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        log_filename = f"debate_log_game{game_id}_{timestamp}.log"
        log_path = os.path.join(self.organized_output_path, log_filename)
        
        # Create logger with game-specific name
        logger_name = f"debate_system_game{game_id}_{timestamp}"
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
        
        # Don't log initialization messages to avoid multiple log files
        # self.logger.info(f"📝 Logging started for Game {game_id}")
        # self.logger.info(f"📁 Log file: {log_path}")
        # self.logger.info(f"🎯 Starting new game - all previous logs are in separate files")
    
    def _ensure_log_file_handler(self):
        """Create the file handler only when we actually need to log content."""
        if hasattr(self, 'log_file_path') and self.log_file_path:
            # Check if we already have a file handler
            has_file_handler = any(isinstance(handler, logging.FileHandler) for handler in self.logger.handlers)
            if not has_file_handler:
                file_handler = logging.FileHandler(self.log_file_path, mode='w', encoding='utf-8')
                file_handler.setLevel(logging.INFO)
                
                # Create formatter
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                file_handler.setFormatter(formatter)
                
                # Add handler to logger
                self.logger.addHandler(file_handler)
    
    def _initialize_agents(self):
        """Initialize all agent clients."""
        for agent_config in self.config.agents:
            if not agent_config.enabled:
                continue
                
            try:
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
                
                self.agents[agent_config.name] = {
                    "client": client,
                    "config": agent_config
                }
                # Don't log initialization messages to avoid multiple log files
                # self.logger.info(f"✓ Initialized {agent_config.name} ({agent_config.provider}/{agent_config.model})")
                
            except Exception as e:
                # Only log errors, not initialization success
                self.logger.error(f"✗ Failed to initialize {agent_config.name}: {e}")
    
    def run_debate_session(self, game: ground_truth) -> DebateSession:
        """Run a complete debate for a single game."""
        # Print current log file to console for user awareness
        print(f"\n🔄 Now running Game {game.game_id}")
        print(f"📁 Current log: {self.organized_output_path}/debate_log_game{game.game_id}_*.log")
        print("=" * 60)
        
        self.logger.info(f"🎯 Starting debate for Game {game.game_id}")
        self.logger.info(f"Players: {[f'Player {i+1}' for i in range(game.num_player)]}")
        
        # Parse ground truth solution
        gt_solution = self._parse_ground_truth_solution(game)
        self.logger.info(f"📋 Ground Truth Solution: {gt_solution}")
        
        # Phase 1: Initial proposals
        self.logger.info("📝 Phase 1: Initial Proposals")
        self.logger.info("=" * 40)
        initial_proposals = self._get_initial_proposals(game)
        
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
        
        # Phase 2: Debate rounds for each player
        self.logger.info("🗣️ Phase 2: Debate Rounds")
        debate_rounds = []
        
        # Get player names from ground truth
        player_names = list(gt_solution.keys())
        
        # Track the current state of all agents (starts with initial proposals)
        current_agent_states = initial_proposals
        
        for player_name in player_names:
            self.logger.info(f"--- Debating {player_name}'s role ---")
            self.logger.info(f"🎯 Ground Truth: {player_name} is a {gt_solution.get(player_name, 'unknown')}")
            debate_round = self._run_debate_round(
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
            
            # DEBUG: Comprehensive summary of all agents' debate responses
            self.logger.info(f"🔍 DEBUG - COMPREHENSIVE DEBATE ROUND {debate_round.round_number} SUMMARY:")
            self.logger.info(f"      {'='*80}")
            for response in debate_round.agent_responses:
                if response.phase == "debate":  # Only show debate phase responses
                    # Get agent config info
                    agent_info = self.agents.get(response.agent_name, {})
                    config = agent_info.get("config", None)
                    provider_model = f"{config.provider}/{config.model}" if config else "unknown/unknown"
                    
                    self.logger.info(f"      🤖 {response.agent_name} ({provider_model}):")
                    self.logger.info(f"          Decision: {response.player_role_assignments}")
                    self.logger.info(f"          Agrees with: {response.agree_with if response.agree_with else 'None'}")
                    self.logger.info(f"          Disagrees with: {response.disagree_with if response.disagree_with else 'None'}")
                    if response.agree_reasoning:
                        self.logger.info(f"          Agree reasoning: {response.agree_reasoning}")
                    if response.disagree_reasoning:
                        self.logger.info(f"          Disagree reasoning: {response.disagree_reasoning}")
                    if response.error:
                        self.logger.warning(f"          ⚠️ ERROR: {response.error}")
                    self.logger.info(f"          {'-'*60}")
            self.logger.info(f"      {'='*80}")
        
        # Phase 3: Final Discussion and Fresh Voting
        self.logger.info("🗣️ Phase 3: Final Discussion and Fresh Voting")
        self.logger.info("=" * 40)
        final_vote = self._conduct_final_discussion_and_vote(game, initial_proposals, debate_rounds)
        
        # Show final vote results
        self.logger.info(f"📊 Final Vote Results:")
        for player, role in final_vote.items():
            correct = "✅" if role == gt_solution.get(player) else "❌"
            self.logger.info(f"  {player}: {role} {correct}")
        
        # Log detailed final vote breakdown
        self.logger.info(f"📊 Final Vote Summary by Player:")
        for player, gt_role in gt_solution.items():
            final_role = final_vote.get(player, "unknown")
            correct = "✅" if final_role == gt_role else "❌"
            self.logger.info(f"  {player}: Final={final_role}, GroundTruth={gt_role} {correct}")
        
        # Phase 4: Supervisor decision if needed
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
                
                # Log detailed supervisor decision breakdown
                self.logger.info(f"📊 Supervisor Decision Summary by Player:")
                for player, gt_role in gt_solution.items():
                    supervisor_role = supervisor_decision.get(player, "unknown")
                    correct = "✅" if supervisor_role == gt_role else "❌"
                    self.logger.info(f"  {player}: Supervisor={supervisor_role}, GroundTruth={gt_role} {correct}")
        
        # Create performance tracking
        performance_tracking = self._create_performance_tracking(
            game, initial_proposals, debate_rounds, final_vote, supervisor_decision
        )
        
        # Create debate
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
        
        # Save debate
        self._save_debate_session(session)
        
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
        
        self.logger.info(f"✅ Debate completed for Game {game.game_id}")
        
        # Generate visualizations automatically
        self._generate_visualizations(session)
        
        return session
    
    def _run_single_game_debate(self, game: ground_truth) -> DebateSession:
        """Run a complete debate for a single game in isolation (for parallel processing)."""
        # Create a temporary debate system instance for this game
        temp_system = MultiAgentDebateSystem(self.debate_config, self.secret_path)
        # Override the parallel processor to use the same instance (shared resources)
        temp_system.parallel_processor = self.parallel_processor
        temp_system.game_parallel_processor = self.game_parallel_processor
        
        # Run the debate session
        return temp_system.run_debate_session(game)
    
    def _get_initial_proposals(self, game: ground_truth) -> List[AgentResponse]:
        """Get initial proposals from all agents."""
        def get_single_proposal(agent_name: str) -> AgentResponse:
            agent_info = self.agents[agent_name]
            client = agent_info["client"]
            config = agent_info["config"]
            
            self.logger.info(f"    🤖 {agent_name} ({config.provider}/{config.model}) getting initial proposal...")
            
            try:
                # Prepare system prompt with confidence if enabled
                system_prompt = get_kks_system_prompt_with_confidence(game.num_player, self.config.self_reported_confidence)
                
                # Make API call based on provider
                if config.provider == "openai":
                    # Check if model supports reasoning parameters
                    if config.model in ["gpt-5-nano", "gpt-5", "gpt-4o"]:
                        response_obj = client.response_completion(
                            user_prompt=game.text_game,
                            system_prompt=system_prompt,
                            model=config.model,
                            reasoning_effort=config.reasoning_effort,
                            verbosity=config.verbosity
                        )
                    else:
                        # Use regular response completion for models that don't support reasoning
                        response_obj = client.response_completion(
                            user_prompt=game.text_game,
                            system_prompt=system_prompt,
                            model=config.model
                        )
                elif config.provider == "gemini":
                    response_obj = client.chat_completion(
                        user_prompt=game.text_game,
                        system_prompt=system_prompt,
                        model=config.model,
                        response_schema=get_kks_response_schema_with_confidence(self.config.self_reported_confidence)
                    )
                else:  # ali, cst
                    response_obj = client.chat_completion(
                        user_prompt=game.text_game,
                        system_prompt=system_prompt,
                        model=config.model
                    )
                
                # Parse response
                player_assignments, explanation, confidence, _, _, _, _ = self._parse_agent_response(response_obj.text)
                
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
            get_single_proposal, agent_names, preserve_order=True
        )
        
        return [p for p in proposals if p is not None]
    
    def _run_debate_round(self, game: ground_truth, player_name: str, 
                         round_num: int, current_agent_states: List[AgentResponse],
                         previous_rounds: List[DebateRound]) -> DebateRound:
        """Run a debate round for a specific player's role."""
        
        # Step 1: Debate period
        self.logger.info(f"  Step 1: Debate period for {player_name}")
        debate_responses = self._conduct_debate_for_player(
            game, player_name, round_num, current_agent_states, previous_rounds
        )
        
        # Step 2: Self-adjustment
        self.logger.info(f"  Step 2: Self-adjustment for {player_name}")
        if self.config.enable_self_adjustment:
            adjusted_responses = self._conduct_self_adjustment(
                game, player_name, round_num, debate_responses, previous_rounds
            )
        else:
            adjusted_responses = debate_responses
        
        # Combine both phases into a single list
        all_responses = debate_responses + adjusted_responses
        
        # Create debate summary
        debate_summary = self._create_debate_summary(adjusted_responses, player_name)
        
        return DebateRound(
            player_name=player_name,
            round_number=round_num,
            agent_responses=all_responses,  # Now includes both debate and self_adjustment phases
            debate_summary=debate_summary,
            consensus_reached=False,  # No intermediate consensus checking
            majority_role=None  # No intermediate majority voting
        )
    
    def _conduct_debate_for_player(self, game: ground_truth, player_name: str,
                                 round_num: int, current_agent_states: List[AgentResponse],
                                 previous_rounds: List[DebateRound]) -> List[AgentResponse]:
        """Conduct debate for a specific player's role."""
        
        def get_debate_response(agent_name: str) -> AgentResponse:
            agent_info = self.agents[agent_name]
            client = agent_info["client"]
            config = agent_info["config"]
            
            self.logger.info(f"      🤖 {agent_name} debating {player_name}...")
            
            try:
                # Create debate prompt
                debate_prompt = self._create_debate_prompt(
                    game, player_name, current_agent_states, previous_rounds
                )
                
                # Make API call
                system_prompt = get_kks_system_prompt_with_confidence(game.num_player, self.config.self_reported_confidence)
                
                if config.provider == "openai":
                    # Check if model supports reasoning parameters
                    if config.model in ["gpt-5-nano", "gpt-5", "gpt-4o"]:
                        response_obj = client.response_completion(
                            user_prompt=debate_prompt,
                            system_prompt=system_prompt,
                            model=config.model,
                            reasoning_effort=config.reasoning_effort,
                            verbosity=config.verbosity
                        )
                    else:
                        # Use regular response completion for models that don't support reasoning
                        response_obj = client.response_completion(
                            user_prompt=debate_prompt,
                            system_prompt=system_prompt,
                            model=config.model
                        )
                elif config.provider == "gemini":
                    # Use debate-specific schema for Gemini
                    debate_response_schema = get_kks_debate_response_schema_with_confidence(self.config.self_reported_confidence)
                    response_obj = client.chat_completion(
                        user_prompt=debate_prompt,
                        system_prompt=system_prompt,
                        model=config.model,
                        response_schema=debate_response_schema
                    )
                else:
                    response_obj = client.chat_completion(
                        user_prompt=debate_prompt,
                        system_prompt=system_prompt,
                        model=config.model
                    )
                
                # Parse response
                player_assignments, explanation, confidence, agree_with, disagree_with, agree_reasoning, disagree_reasoning = self._parse_agent_response(response_obj.text, "debate")
                
                # DEBUG: Log raw response for debugging
                self.logger.info(f"      🔍 DEBUG - {agent_name} RAW DEBATE RESPONSE:")
                self.logger.info(f"      {'='*60}")
                self.logger.info(f"      Raw text: {response_obj.text}")
                self.logger.info(f"      {'='*60}")
                
                self.logger.info(f"      ✅ {agent_name} debate completed")
                if self.config.self_reported_confidence:
                    self.logger.info(f"      📊 {agent_name} confidence: {confidence}")
                else:
                    self.logger.info(f"      📊 {agent_name} confidence: not requested (SRC disabled)")
                
                # Log agreement/disagreement analysis with warnings
                self.logger.info(f"      🔍 DEBUG - {agent_name} AGREEMENT/DISAGREEMENT ANALYSIS:")
                if agree_with and len(agree_with) > 0:
                    self.logger.info(f"      🤝 {agent_name} agrees with: {', '.join(agree_with)}")
                    if agree_reasoning and agree_reasoning.strip():
                        self.logger.info(f"      💭 Agree reasoning: {agree_reasoning}")
                    else:
                        self.logger.warning(f"      ⚠️ WARNING: {agent_name} has no agree_reasoning despite agreeing with agents")
                else:
                    self.logger.warning(f"      ⚠️ WARNING: {agent_name} has no agree_with information")
                
                if disagree_with and len(disagree_with) > 0:
                    self.logger.info(f"      ❌ {agent_name} disagrees with: {', '.join(disagree_with)}")
                    if disagree_reasoning and disagree_reasoning.strip():
                        self.logger.info(f"      💭 Disagree reasoning: {disagree_reasoning}")
                    else:
                        self.logger.warning(f"      ⚠️ WARNING: {agent_name} has no disagree_reasoning despite disagreeing with agents")
                else:
                    self.logger.warning(f"      ⚠️ WARNING: {agent_name} has no disagree_with information")
                
                # Log final decision
                if player_assignments:
                    for player, role in player_assignments.items():
                        self.logger.info(f"      🎯 {agent_name} final decision: {player} is a {role}")
                else:
                    self.logger.warning(f"      ⚠️ WARNING: {agent_name} has no player role assignments")
                
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
                self.logger.warning(f"      ⚠️ WARNING: {agent_name} debate phase failed - no raw output available")
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
            get_debate_response, agent_names, preserve_order=True
        )
        
        return [r for r in responses if r is not None]
    
    def _conduct_self_adjustment(self, game: ground_truth, player_name: str,
                               round_num: int, debate_responses: List[AgentResponse],
                               previous_rounds: List[DebateRound]) -> List[AgentResponse]:
        """Conduct self-adjustment phase for agents."""
        
        def get_adjustment_response(agent_name: str) -> AgentResponse:
            agent_info = self.agents[agent_name]
            client = agent_info["client"]
            config = agent_info["config"]
            
            self.logger.info(f"      🔄 {agent_name} self-adjusting for {player_name}...")
            
            try:
                # Create self-adjustment prompt
                adjustment_prompt = self._create_self_adjustment_prompt(
                    game, player_name, debate_responses, previous_rounds
                )
                
                # Make API call
                system_prompt = get_kks_system_prompt_with_confidence(game.num_player, self.config.self_reported_confidence)
                response_schema = get_kks_response_schema_with_confidence(self.config.self_reported_confidence)
                
                if config.provider == "openai":
                    # Check if model supports reasoning parameters
                    if config.model in ["gpt-5-nano", "gpt-5", "gpt-4o"]:
                        response_obj = client.response_completion(
                            user_prompt=adjustment_prompt,
                            system_prompt=system_prompt,
                            model=config.model,
                            reasoning_effort=config.reasoning_effort,
                            verbosity=config.verbosity
                        )
                    else:
                        # Use regular response completion for models that don't support reasoning
                        response_obj = client.response_completion(
                            user_prompt=adjustment_prompt,
                            system_prompt=system_prompt,
                            model=config.model
                        )
                elif config.provider == "gemini":
                    response_obj = client.chat_completion(
                        user_prompt=adjustment_prompt,
                        system_prompt=system_prompt,
                        model=config.model,
                        response_schema=response_schema
                    )
                else:
                    response_obj = client.chat_completion(
                        user_prompt=adjustment_prompt,
                        system_prompt=system_prompt,
                        model=config.model
                    )
                
                # Parse response
                player_assignments, explanation, confidence, _, _, _, _ = self._parse_agent_response(response_obj.text, "self_adjustment")
                
                self.logger.info(f"      ✅ {agent_name} self-adjustment completed")
                if self.config.self_reported_confidence:
                    self.logger.info(f"      📊 {agent_name} confidence: {confidence}")
                else:
                    self.logger.info(f"      📊 {agent_name} confidence: not requested (SRC disabled)")
                
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
            get_adjustment_response, agent_names, preserve_order=True
        )
        
        return [r for r in responses if r is not None]
    
    def _conduct_final_discussion_and_vote(self, game: ground_truth, initial_proposals: List[AgentResponse], debate_rounds: List[DebateRound]) -> Dict[str, str]:
        """Conduct final discussion where agents reconsider everything and make fresh decisions."""
        self.logger.info("Conducting final discussion and fresh voting...")
        
        # Get all player names
        player_names = list(initial_proposals[0].player_role_assignments.keys())
        
        # Create final discussion prompt
        final_discussion_prompt = self._create_final_discussion_prompt(game, initial_proposals, debate_rounds)
        
        # Get fresh votes from all agents
        fresh_votes = []
        agent_names = [name for name in self.agents.keys()]
        
        for agent_name in agent_names:
            agent_info = self.agents[agent_name]
            client = agent_info["client"]
            config = agent_info["config"]
            
            self.logger.info(f"  🤖 {agent_name} making final decision...")
            
            try:
                # Make API call for final decision
                system_prompt = get_kks_system_prompt_with_confidence(game.num_player, self.config.self_reported_confidence)
                response_schema = get_kks_response_schema_with_confidence(self.config.self_reported_confidence)
                
                if config.provider == "openai":
                    if config.model in ["gpt-5-nano", "gpt-5", "gpt-4o"]:
                        response_obj = client.response_completion(
                            user_prompt=final_discussion_prompt,
                            system_prompt=system_prompt,
                            model=config.model,
                            reasoning_effort=config.reasoning_effort,
                            verbosity=config.verbosity
                        )
                    else:
                        response_obj = client.response_completion(
                            user_prompt=final_discussion_prompt,
                            system_prompt=system_prompt,
                            model=config.model
                        )
                elif config.provider == "gemini":
                    response_obj = client.chat_completion(
                        user_prompt=final_discussion_prompt,
                        system_prompt=system_prompt,
                        model=config.model,
                        response_schema=response_schema
                    )
                else:
                    response_obj = client.chat_completion(
                        user_prompt=final_discussion_prompt,
                        system_prompt=system_prompt,
                        model=config.model
                    )
                
                # Parse response
                player_assignments, explanation, confidence, _, _, _, _ = self._parse_agent_response(response_obj.text)
                
                fresh_votes.append({
                    "agent_name": agent_name,
                    "player_assignments": player_assignments,
                    "explanation": explanation,
                    "confidence": confidence
                })
                
                self.logger.info(f"  ✅ {agent_name} final decision completed")
                if self.config.self_reported_confidence:
                    self.logger.info(f"  📊 {agent_name} confidence: {confidence}")
                else:
                    self.logger.info(f"  📊 {agent_name} confidence: not requested (SRC disabled)")
                
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
    
    def _create_final_discussion_prompt(self, game: ground_truth, initial_proposals: List[AgentResponse], debate_rounds: List[DebateRound]) -> str:
        """Create final discussion prompt for agents to reconsider everything."""
        
        prompt = f"""You are participating in the FINAL DISCUSSION phase of a multi-agent debate about a Knight-Knaves-Spy game.

GAME INFORMATION:
{game.text_game}

DEBATE HISTORY SUMMARY:
We have completed individual debates for each player's role. Now it's time for the final discussion where you should reconsider everything and make your final decision.

INITIAL PROPOSALS:
"""
        
        for proposal in initial_proposals:
            confidence_info = ""
            if self.config.self_reported_confidence and proposal.confidence > 0:
                confidence_info = f" (confidence: {proposal.confidence})"
            prompt += f"\n{proposal.agent_name} initially thought: {proposal.player_role_assignments}{confidence_info}"
            prompt += f"\nTheir reasoning: {proposal.explanation[:200]}...\n"
        
        prompt += "\nDEBATE ROUNDS SUMMARY:\n"
        for round_data in debate_rounds:
            prompt += f"\n--- Round {round_data.round_number}: {round_data.player_name} ---\n"
            for response in round_data.agent_responses:
                role = response.player_role_assignments.get(round_data.player_name, "unknown")
                confidence_info = ""
                if self.config.self_reported_confidence and response.confidence > 0:
                    confidence_info = f" (confidence: {response.confidence})"
                prompt += f"{response.agent_name} thought {round_data.player_name} is a {role}{confidence_info}\n"
                
                # Include agreement/disagreement information if available (for debate phase responses)
                if response.phase == "debate":
                    if response.agree_with and len(response.agree_with) > 0:
                        prompt += f"  Agreed with: {', '.join(response.agree_with)}"
                        if response.agree_reasoning and response.agree_reasoning.strip():
                            prompt += f" - Reasoning: {response.agree_reasoning[:100]}..."
                        prompt += "\n"
                    
                    if response.disagree_with and len(response.disagree_with) > 0:
                        prompt += f"  Disagreed with: {', '.join(response.disagree_with)}"
                        if response.disagree_reasoning and response.disagree_reasoning.strip():
                            prompt += f" - Reasoning: {response.disagree_reasoning[:100]}..."
                        prompt += "\n"
                else:
                    # Include explanation for non-debate phases (initial, self_adjustment)
                    if response.explanation:
                        prompt += f"Reasoning: {response.explanation[:150]}...\n"
        
        prompt += f"""

FINAL DISCUSSION INSTRUCTIONS:
Now that you have seen the complete debate history, please make your FINAL decision for ALL players. Consider:
1. All the arguments made during the debates
2. How your thinking may have evolved
3. Any new insights from the discussion
4. The overall consistency of the solution

This is your final chance to make the best possible decision. Think carefully and provide your complete analysis.

Return your response in the same JSON format:
{{
    "players": [
        {{"name": "player_name", "role": "role"}},
        ...
    ],
    "explanation": "Your final comprehensive reasoning after considering the entire debate process"
}}"""
        
        return prompt
    
    def _get_supervisor_decision(self, game: ground_truth, initial_proposals: List[AgentResponse], debate_rounds: List[DebateRound]) -> Dict[str, str]:
        """Get supervisor decision when consensus cannot be reached."""
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
            
            # Create supervisor prompt
            supervisor_prompt = self._create_supervisor_prompt(game, initial_proposals, debate_rounds)
            
            # Make API call
            system_prompt = get_kks_system_prompt_with_confidence(game.num_player, self.config.self_reported_confidence)
            response_schema = get_kks_response_schema_with_confidence(self.config.self_reported_confidence)
            
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
    
    def _parse_agent_response(self, response_text: str, phase: str = "initial") -> Tuple[Dict[str, str], str, float, Optional[List[str]], Optional[List[str]], Optional[str], Optional[str]]:
        """Parse agent response to extract player assignments, explanation, confidence, and debate analysis."""
        try:
            # First try to extract JSON from the response (handle extra text after JSON)
            json_text = self._extract_json_from_response(response_text)
            response_data = json.loads(json_text)
            
            # Extract player assignments
            player_assignments = {}
            explanation = ""
            confidence = 5.0  # Default to medium confidence (1-10 scale)
            agree_with = None
            disagree_with = None
            agree_reasoning = None
            disagree_reasoning = None
            
            if phase == "debate":
                # Debate phase format: single player role with agreement/disagreement analysis
                if "player_role" in response_data and "role" in response_data:
                    player_assignments[response_data["player_role"]] = response_data["role"]
                
                # Extract agreement/disagreement information
                agree_with = response_data.get("agree_with", [])
                disagree_with = response_data.get("disagree_with", [])
                agree_reasoning = self._clean_explanation(response_data.get("agree_reasoning", ""))
                disagree_reasoning = self._clean_explanation(response_data.get("disagree_reasoning", ""))
                
                # For debate phase, we don't use explanation field anymore
                explanation = ""
                confidence = self._parse_confidence(response_data.get("confidence", 5))
            elif "players" in response_data and isinstance(response_data["players"], list):
                # New format with players array
                for player_data in response_data["players"]:
                    if isinstance(player_data, dict) and "name" in player_data and "role" in player_data:
                        player_assignments[player_data["name"]] = player_data["role"]
                explanation = self._clean_explanation(response_data.get("explanation", ""))
                confidence = self._parse_confidence(response_data.get("confidence", 5))
            else:
                # Old format - direct name: role mapping
                for key, value in response_data.items():
                    if key not in ["explanation", "confidence", "agree_with", "disagree_with", "agree_reasoning", "disagree_reasoning"] and isinstance(value, str):
                        player_assignments[key] = value
                explanation = self._clean_explanation(response_data.get("explanation", ""))
                confidence = self._parse_confidence(response_data.get("confidence", 5))
            
            # If confidence not found in JSON, try to extract from extra text
            if confidence == 5.0 and "confidence" not in response_data:
                confidence = self._extract_confidence_from_text(response_text)
            
            return player_assignments, explanation, confidence, agree_with, disagree_with, agree_reasoning, disagree_reasoning
            
        except (json.JSONDecodeError, ValueError):
            # If JSON parsing fails, try to extract using regex
            import re
            pattern = r'(\w+)\s+is\s+a\s+(knight|knave|spy)\.'
            matches = re.findall(pattern, response_text, re.IGNORECASE)
            
            player_assignments = {}
            for player_name, role in matches:
                player_assignments[player_name] = role.lower()
            
            # Try to extract confidence from text even in regex fallback
            confidence = self._extract_confidence_from_text(response_text)
            
            # For regex fallback, return None for debate-specific fields
            return player_assignments, self._clean_explanation(response_text), confidence, None, None, None, None
    
    def _extract_json_from_response(self, response_text: str) -> str:
        """Extract JSON from response text, handling extra text after JSON."""
        import re
        
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
    
    def _extract_confidence_from_text(self, response_text: str) -> float:
        """Extract confidence value from text patterns like 'Confidence: 5'."""
        import re
        
        # Try different patterns for confidence
        patterns = [
            r'confidence:\s*(\d+)',  # "Confidence: 5"
            r'my confidence is\s*(\d+)',  # "My confidence is 5"
            r'confidence level:\s*(\d+)',  # "Confidence level: 5"
            r'confidence\s*=\s*(\d+)',  # "confidence = 5"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                try:
                    confidence = float(match.group(1))
                    return self._parse_confidence(confidence)
                except (ValueError, TypeError):
                    continue
        
        # If no confidence found, return default
        return 5.0
    
    def _clean_explanation(self, explanation: str) -> str:
        """Clean explanation text to remove extra formatting and keep only the core content."""
        if not explanation:
            return ""
        
        # Remove common prefixes/suffixes that models sometimes add
        explanation = explanation.strip()
        
        # Remove "Explanation:" prefix if present
        if explanation.lower().startswith("explanation:"):
            explanation = explanation[12:].strip()
        
        # Remove "Reasoning:" prefix if present
        if explanation.lower().startswith("reasoning:"):
            explanation = explanation[10:].strip()
        
        # Remove confidence statements that might be mixed in
        import re
        explanation = re.sub(r'confidence:\s*\d+', '', explanation, flags=re.IGNORECASE)
        explanation = re.sub(r'my confidence is\s*\d+', '', explanation, flags=re.IGNORECASE)
        
        # Clean up extra whitespace
        explanation = re.sub(r'\s+', ' ', explanation).strip()
        
        return explanation
    
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
        """Parse ground truth solution to extract player-role pairs."""
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
            self.logger.error(f"Error parsing ground truth: {e}")
            return {}
    
    def _create_debate_prompt(self, game: ground_truth, player_name: str,
                            current_agent_states: List[AgentResponse],
                            previous_rounds: List[DebateRound]) -> str:
        """Create debate prompt for a specific player's role."""
        
        prompt = f"""You are participating in a multi-agent debate about a Knight-Knaves-Spy game.

GAME INFORMATION:
{game.text_game}

CURRENT FOCUS: We are debating the role of {player_name}.

OTHER AGENTS' CURRENT POSITIONS (from last step):
"""
        
        for agent_state in current_agent_states:
            confidence_info = ""
            if self.config.self_reported_confidence and agent_state.confidence > 0:
                confidence_info = f" (confidence: {agent_state.confidence})"
            prompt += f"\n{agent_state.agent_name} thinks {player_name} is a {agent_state.player_role_assignments.get(player_name, 'unknown')}.{confidence_info}"
            prompt += f" Their reasoning: {agent_state.explanation}\n"
        
        if previous_rounds:
            prompt += "\nPREVIOUS DEBATE ROUNDS:\n"
            for round_data in previous_rounds:
                prompt += f"\nRound {round_data.round_number} - {round_data.player_name}:\n"
                for response in round_data.agent_responses:
                    if round_data.player_name in response.player_role_assignments:
                        confidence_info = ""
                        if self.config.self_reported_confidence and response.confidence > 0:
                            confidence_info = f" (confidence: {response.confidence})"
                        prompt += f"  {response.agent_name}: {round_data.player_name} is a {response.player_role_assignments[round_data.player_name]}{confidence_info}\n"
        
        prompt += f"""

Please provide your debate analysis focusing specifically on {player_name}'s role. Consider the other agents' arguments and provide your reasoning.

IMPORTANT: For this debate phase, you should:
1. Analyze which agents you agree with and which you disagree with regarding {player_name}'s role
2. Provide reasoning for your agreements and disagreements
3. Make your final decision on {player_name}'s role
4. You will have a chance to provide your complete solution in the self-adjustment phase

Return your response in the following JSON format:
{{
    "player_role": "{player_name}",
    "role": "knight/knave/spy",
    "agree_with": ["agent_name1", "agent_name2"],
    "disagree_with": ["agent_name3"],
    "agree_reasoning": "Your reasoning for why you agree with the specified agents",
    "disagree_reasoning": "Your reasoning for why you disagree with the specified agents"
}}

IMPORTANT: Keep your reasoning concise and focused (aim for under 50 words) for both agree_reasoning and disagree_reasoning fields."""
        
        return prompt
    
    def _create_self_adjustment_prompt(self, game: ground_truth, player_name: str,
                                     debate_responses: List[AgentResponse],
                                     previous_rounds: List[DebateRound]) -> str:
        """Create self-adjustment prompt for agents."""
        
        prompt = f"""You are in the self-adjustment phase of a multi-agent debate about a Knight-Knaves-Spy game.

GAME INFORMATION:
{game.text_game}

CURRENT FOCUS: We just finished debating {player_name}'s role.

DEBATE SUMMARY:
"""
        
        for response in debate_responses:
            confidence_info = ""
            if self.config.self_reported_confidence and response.confidence > 0:
                confidence_info = f" (confidence: {response.confidence})"
            prompt += f"\n{response.agent_name} thinks {player_name} is a {response.player_role_assignments.get(player_name, 'unknown')}.{confidence_info}"
            
            # Include agreement/disagreement information if available
            if response.agree_with and len(response.agree_with) > 0:
                prompt += f" {response.agent_name} agrees with: {', '.join(response.agree_with)}"
                if response.agree_reasoning and response.agree_reasoning.strip():
                    prompt += f" - Reasoning: {response.agree_reasoning}"
                prompt += "\n"
            
            if response.disagree_with and len(response.disagree_with) > 0:
                prompt += f" {response.agent_name} disagrees with: {', '.join(response.disagree_with)}"
                if response.disagree_reasoning and response.disagree_reasoning.strip():
                    prompt += f" - Reasoning: {response.disagree_reasoning}"
                prompt += "\n"
            
            # Include explanation if available and no agreement/disagreement info was provided
            if response.explanation and not (response.agree_with or response.disagree_with):
                prompt += f" Their reasoning: {response.explanation}\n"
        
        prompt += f"""

Based on the debate, please provide your final assessment. You may adjust your position on {player_name} or any other players if you've been convinced by the arguments.

IMPORTANT: For this self-adjustment phase, provide your complete solution for ALL players, not just {player_name}.

Return your response in the following JSON format:
{{
    "players": [
        {{"name": "player_name", "role": "role"}},
        ...
    ],
    "explanation": "Your final reasoning after considering the debate"
}}

IMPORTANT: Keep your explanation concise and focused (aim for under 50 words)."""
        
        return prompt
    
    def _create_supervisor_prompt(self, game: ground_truth, initial_proposals: List[AgentResponse], debate_rounds: List[DebateRound]) -> str:
        """Create supervisor prompt for final decision."""
        
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

IMPORTANT: Keep your explanation concise and focused (aim for under 50 words)."""
        
        return prompt
    
    def _check_consensus(self, responses: List[AgentResponse], player_name: str) -> Optional[str]:
        """Check if there's consensus on a player's role."""
        if not responses:
            return None
        
        votes = []
        for response in responses:
            if player_name in response.player_role_assignments:
                votes.append(response.player_role_assignments[player_name])
        
        if not votes:
            return None
        
        from collections import Counter
        vote_counts = Counter(votes)
        most_common = vote_counts.most_common(1)[0]
        
        # Check if there's a clear majority (more than 50%)
        if most_common[1] > len(votes) / 2:
            return most_common[0]
        
        return None
    
    def _create_debate_summary(self, responses: List[AgentResponse], player_name: str) -> str:
        """Create a summary of the debate for a player."""
        summary = f"Debate summary for {player_name}:\n"
        
        for response in responses:
            role = response.player_role_assignments.get(player_name, "unknown")
            summary += f"- {response.agent_name}: {role}\n"
        
        return summary
    
    def _is_consensus_reached(self, final_vote: Dict[str, str]) -> bool:
        """Check if consensus was reached in final vote."""
        return len(final_vote) > 0
    
    def _create_performance_tracking(self, game: ground_truth, initial_proposals: List[AgentResponse],
                                   debate_rounds: List[DebateRound], final_vote: Dict[str, str],
                                   supervisor_decision: Optional[Dict[str, str]]) -> Dict[str, Any]:
        """Create performance tracking data."""
        
        gt_solution = self._parse_ground_truth_solution(game)
        
        tracking = {
            "game_id": game.game_id,
            "ground_truth": gt_solution,
            "initial_accuracy": {},
            "final_accuracy": {},
            "consensus_accuracy": {},
            "agent_performance": {},
            "round_by_round": [],
            "detailed_per_player_tracking": {
                "initial": {},
                "rounds": {},
                "final": {},
                "supervisor": {}
            }
        }
        
        # Calculate initial accuracy for each agent and detailed per-player tracking
        for proposal in initial_proposals:
            correct = 0
            total = len(gt_solution)
            tracking["detailed_per_player_tracking"]["initial"][proposal.agent_name] = {}
            
            for player, role in gt_solution.items():
                predicted_role = proposal.player_role_assignments.get(player, "unknown")
                is_correct = predicted_role == role
                if is_correct:
                    correct += 1
                tracking["detailed_per_player_tracking"]["initial"][proposal.agent_name][player] = {
                    "predicted": predicted_role,
                    "ground_truth": role,
                    "correct": is_correct
                }
            tracking["initial_accuracy"][proposal.agent_name] = correct / total if total > 0 else 0
        
        # Calculate final accuracy and detailed tracking
        if final_vote:
            correct = 0
            total = len(gt_solution)
            tracking["detailed_per_player_tracking"]["final"]["majority_vote"] = {}
            
            for player, role in gt_solution.items():
                predicted_role = final_vote.get(player, "unknown")
                is_correct = predicted_role == role
                if is_correct:
                    correct += 1
                tracking["detailed_per_player_tracking"]["final"]["majority_vote"][player] = {
                    "predicted": predicted_role,
                    "ground_truth": role,
                    "correct": is_correct
                }
            tracking["final_accuracy"]["majority_vote"] = correct / total if total > 0 else 0
        
        if supervisor_decision:
            correct = 0
            total = len(gt_solution)
            tracking["detailed_per_player_tracking"]["supervisor"]["supervisor"] = {}
            
            for player, role in gt_solution.items():
                predicted_role = supervisor_decision.get(player, "unknown")
                is_correct = predicted_role == role
                if is_correct:
                    correct += 1
                tracking["detailed_per_player_tracking"]["supervisor"]["supervisor"][player] = {
                    "predicted": predicted_role,
                    "ground_truth": role,
                    "correct": is_correct
                }
            tracking["final_accuracy"]["supervisor"] = correct / total if total > 0 else 0
        
        # Round-by-round tracking with detailed per-player data
        for round_data in debate_rounds:
            round_tracking = {
                "player": round_data.player_name,
                "round": round_data.round_number,
                "agent_accuracy": {}
            }
            
            # Initialize round tracking for this round
            tracking["detailed_per_player_tracking"]["rounds"][f"round_{round_data.round_number}"] = {
                "player_focus": round_data.player_name,
                "agents": {}
            }
            
            for response in round_data.agent_responses:
                correct = 0
                total = len(gt_solution)
                tracking["detailed_per_player_tracking"]["rounds"][f"round_{round_data.round_number}"]["agents"][response.agent_name] = {}
                
                for player, role in gt_solution.items():
                    predicted_role = response.player_role_assignments.get(player, "unknown")
                    is_correct = predicted_role == role
                    if is_correct:
                        correct += 1
                    
                    tracking["detailed_per_player_tracking"]["rounds"][f"round_{round_data.round_number}"]["agents"][response.agent_name][player] = {
                        "predicted": predicted_role,
                        "ground_truth": role,
                        "correct": is_correct
                    }
                round_tracking["agent_accuracy"][response.agent_name] = correct / total if total > 0 else 0
            
            tracking["round_by_round"].append(round_tracking)
        
        return tracking
    
    def _save_debate_session(self, session: DebateSession):
        """Save debate to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"debate_{session.game_id}_{timestamp}.json"
        filepath = os.path.join(self.organized_output_path, filename)
        
        # Convert to serializable format
        session_dict = asdict(session)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(session_dict, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"💾 Debate saved to: {filepath}")
    
    def run_batch_debate(self, games: List[ground_truth]) -> List[DebateSession]:
        """Run debates for multiple games sequentially."""
        sessions = []
        
        for i, game in enumerate(games):
            # Setup logging for this specific game first
            self._setup_logging(game.game_id)
            
            # Now log the batch processing message to the correct log file
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Running debate {i+1}/{len(games)}")
            self.logger.info(f"{'='*60}")
            
            session = self.run_debate_session(game)
            sessions.append(session)
        
        return sessions
    
    def run_parallel_batch_debate(self, games: List[ground_truth]) -> List[DebateSession]:
        """Run debates for multiple games in parallel."""
        print(f"🚀 Starting parallel processing of {len(games)} games")
        print(f"🔧 Using {self.game_parallel_processor.num_workers} workers for game-level parallelism")
        print(f"🤖 Each game will use {self.parallel_processor.num_workers} workers for agent-level parallelism")
        
        # Create a wrapper function that handles logging setup for each game
        def run_game_with_logging(game: ground_truth) -> DebateSession:
            # Create a temporary system instance for this specific game without logging
            temp_system = MultiAgentDebateSystem(self.debate_config, self.secret_path, setup_logging=False)
            # Use the same parallel processors (shared resources)
            temp_system.parallel_processor = self.parallel_processor
            temp_system.game_parallel_processor = self.game_parallel_processor
            
            # Setup logging for this specific game (this will prepare for log file creation)
            temp_system._setup_logging(game.game_id, create_log_file=True)
            
            # Ensure the log file handler is created before logging
            temp_system._ensure_log_file_handler()
            
            # Log the parallel processing message
            temp_system.logger.info(f"\n{'='*60}")
            temp_system.logger.info(f"Running parallel debate for Game {game.game_id}")
            temp_system.logger.info(f"{'='*60}")
            
            # Run the debate session
            return temp_system.run_debate_session(game)
        
        # Process games in parallel
        sessions = self.game_parallel_processor.process_tasks(
            run_game_with_logging, 
            games, 
            preserve_order=True
        )
        
        print(f"✅ Completed parallel processing of {len(sessions)} games")
        return sessions
    
    def _generate_visualizations(self, session: DebateSession):
        """Generate visualizations for a completed debate session."""
        if not VISUALIZER_AVAILABLE:
            self.logger.warning("📊 Visualizer not available - skipping visualization generation")
            return
        
        try:
            self.logger.info("📊 Generating visualizations...")
            
            # Create visualizer instance
            visualizer = DebateVisualizer(output_path=self.organized_output_path)
            
            # Generate all visualizations
            results = visualizer.create_all_visualizations([session])
            
            # Log the generated files
            self.logger.info("📈 Generated visualizations:")
            for name, path in results.items():
                self.logger.info(f"  - {name}: {os.path.basename(path)}")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to generate visualizations: {e}")
            # Don't raise the exception - visualization failure shouldn't stop the debate
