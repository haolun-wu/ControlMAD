import json
import os
import time
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from project_types import ground_truth, response_format, token_usage
from debate_config import DebateConfig, AgentConfig
from utility import openai_client, gemini_client, ali_client, cstcloud, ParallelProcessor
from prompts import kks_system_prompt, kks_response_schema

@dataclass
class AgentResponse:
    """Response from a single agent."""
    agent_name: str
    game_id: int
    round_number: int
    phase: str  # "initial", "debate", "self_adjustment", "final"
    player_role_assignments: Dict[str, str]  # {player_name: role}
    explanation: str
    confidence: float = 0.0
    response_obj: Optional[response_format] = None
    timestamp: str = ""

@dataclass
class DebateRound:
    """A single round of debate for a specific player's role."""
    player_name: str
    round_number: int
    agent_responses: List[AgentResponse]
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
    """Multi-agent debate system for knight-knaves-spy games."""
    
    def __init__(self, debate_config: DebateConfig, secret_path: str = "secret.json"):
        self.config = debate_config
        self.secret_path = secret_path
        self.agents = {}
        self.parallel_processor = ParallelProcessor(num_workers=len(debate_config.agents))
        
        # Initialize agent clients
        self._initialize_agents()
        
        # Create output directory
        os.makedirs(debate_config.output_path, exist_ok=True)
    
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
                print(f"✓ Initialized {agent_config.name} ({agent_config.provider}/{agent_config.model})")
                
            except Exception as e:
                print(f"✗ Failed to initialize {agent_config.name}: {e}")
    
    def run_debate_session(self, game: ground_truth) -> DebateSession:
        """Run a complete debate session for a single game."""
        print(f"\n🎯 Starting debate session for Game {game.game_id}")
        print(f"Players: {[f'Player {i+1}' for i in range(game.num_player)]}")
        
        # Parse ground truth solution
        gt_solution = self._parse_ground_truth_solution(game)
        
        # Phase 1: Initial proposals
        print("\n📝 Phase 1: Initial Proposals")
        initial_proposals = self._get_initial_proposals(game)
        
        # Phase 2: Debate rounds for each player
        print("\n🗣️ Phase 2: Debate Rounds")
        debate_rounds = []
        
        # Get player names from ground truth
        player_names = list(gt_solution.keys())
        
        for player_name in player_names:
            print(f"\n--- Debating {player_name}'s role ---")
            debate_round = self._run_debate_round(
                game, player_name, len(debate_rounds) + 1, 
                initial_proposals, debate_rounds
            )
            debate_rounds.append(debate_round)
        
        # Phase 3: Final majority vote
        print("\n🗳️ Phase 3: Final Majority Vote")
        final_vote = self._conduct_final_vote(game, debate_rounds)
        
        # Phase 4: Supervisor decision if needed
        supervisor_decision = None
        if not self._is_consensus_reached(final_vote):
            print("\n👨‍💼 Phase 4: Supervisor Decision")
            supervisor_decision = self._get_supervisor_decision(game, debate_rounds)
        
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
        
        # Save session
        self._save_debate_session(session)
        
        return session
    
    def _get_initial_proposals(self, game: ground_truth) -> List[AgentResponse]:
        """Get initial proposals from all agents."""
        def get_single_proposal(agent_name: str) -> AgentResponse:
            agent_info = self.agents[agent_name]
            client = agent_info["client"]
            config = agent_info["config"]
            
            try:
                # Prepare system prompt
                system_prompt = kks_system_prompt.replace("{num_player}", str(game.num_player))
                
                # Make API call based on provider
                if config.provider == "openai":
                    response_obj = client.response_completion(
                        user_prompt=game.text_game,
                        system_prompt=system_prompt,
                        model=config.model,
                        reasoning_effort=config.reasoning_effort,
                        verbosity=config.verbosity
                    )
                elif config.provider == "gemini":
                    response_obj = client.chat_completion(
                        user_prompt=game.text_game,
                        system_prompt=system_prompt,
                        model=config.model,
                        response_schema=kks_response_schema
                    )
                else:  # ali, cst
                    response_obj = client.chat_completion(
                        user_prompt=game.text_game,
                        system_prompt=system_prompt,
                        model=config.model
                    )
                
                # Parse response
                player_assignments, explanation = self._parse_agent_response(response_obj.text)
                
                return AgentResponse(
                    agent_name=agent_name,
                    game_id=game.game_id,
                    round_number=0,
                    phase="initial",
                    player_role_assignments=player_assignments,
                    explanation=explanation,
                    response_obj=response_obj,
                    timestamp=datetime.now().isoformat()
                )
                
            except Exception as e:
                print(f"Error getting proposal from {agent_name}: {e}")
                return AgentResponse(
                    agent_name=agent_name,
                    game_id=game.game_id,
                    round_number=0,
                    phase="initial",
                    player_role_assignments={},
                    explanation=f"Error: {str(e)}",
                    timestamp=datetime.now().isoformat()
                )
        
        # Get proposals from all agents in parallel
        agent_names = [name for name in self.agents.keys()]
        proposals = self.parallel_processor.process_tasks(
            get_single_proposal, agent_names, preserve_order=True
        )
        
        return [p for p in proposals if p is not None]
    
    def _run_debate_round(self, game: ground_truth, player_name: str, 
                         round_num: int, initial_proposals: List[AgentResponse],
                         previous_rounds: List[DebateRound]) -> DebateRound:
        """Run a debate round for a specific player's role."""
        
        # Step 1: Debate period
        print(f"  Step 1: Debate period for {player_name}")
        debate_responses = self._conduct_debate_for_player(
            game, player_name, round_num, initial_proposals, previous_rounds
        )
        
        # Step 2: Self-adjustment
        print(f"  Step 2: Self-adjustment for {player_name}")
        if self.config.enable_self_adjustment:
            adjusted_responses = self._conduct_self_adjustment(
                game, player_name, round_num, debate_responses, previous_rounds
            )
        else:
            adjusted_responses = debate_responses
        
        # Step 3: Check consensus
        consensus_role = self._check_consensus(adjusted_responses, player_name)
        consensus_reached = consensus_role is not None
        
        # Create debate summary
        debate_summary = self._create_debate_summary(adjusted_responses, player_name)
        
        return DebateRound(
            player_name=player_name,
            round_number=round_num,
            agent_responses=adjusted_responses,
            debate_summary=debate_summary,
            consensus_reached=consensus_reached,
            majority_role=consensus_role
        )
    
    def _conduct_debate_for_player(self, game: ground_truth, player_name: str,
                                 round_num: int, initial_proposals: List[AgentResponse],
                                 previous_rounds: List[DebateRound]) -> List[AgentResponse]:
        """Conduct debate for a specific player's role."""
        
        def get_debate_response(agent_name: str) -> AgentResponse:
            agent_info = self.agents[agent_name]
            client = agent_info["client"]
            config = agent_info["config"]
            
            try:
                # Create debate prompt
                debate_prompt = self._create_debate_prompt(
                    game, player_name, initial_proposals, previous_rounds
                )
                
                # Make API call
                if config.provider == "openai":
                    response_obj = client.response_completion(
                        user_prompt=debate_prompt,
                        system_prompt=kks_system_prompt.replace("{num_player}", str(game.num_player)),
                        model=config.model,
                        reasoning_effort=config.reasoning_effort,
                        verbosity=config.verbosity
                    )
                elif config.provider == "gemini":
                    response_obj = client.chat_completion(
                        user_prompt=debate_prompt,
                        system_prompt=kks_system_prompt.replace("{num_player}", str(game.num_player)),
                        model=config.model,
                        response_schema=kks_response_schema
                    )
                else:
                    response_obj = client.chat_completion(
                        user_prompt=debate_prompt,
                        system_prompt=kks_system_prompt.replace("{num_player}", str(game.num_player)),
                        model=config.model
                    )
                
                # Parse response
                player_assignments, explanation = self._parse_agent_response(response_obj.text)
                
                return AgentResponse(
                    agent_name=agent_name,
                    game_id=game.game_id,
                    round_number=round_num,
                    phase="debate",
                    player_role_assignments=player_assignments,
                    explanation=explanation,
                    response_obj=response_obj,
                    timestamp=datetime.now().isoformat()
                )
                
            except Exception as e:
                print(f"Error in debate for {agent_name}: {e}")
                return AgentResponse(
                    agent_name=agent_name,
                    game_id=game.game_id,
                    round_number=round_num,
                    phase="debate",
                    player_role_assignments={},
                    explanation=f"Error: {str(e)}",
                    timestamp=datetime.now().isoformat()
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
            
            try:
                # Create self-adjustment prompt
                adjustment_prompt = self._create_self_adjustment_prompt(
                    game, player_name, debate_responses, previous_rounds
                )
                
                # Make API call
                if config.provider == "openai":
                    response_obj = client.response_completion(
                        user_prompt=adjustment_prompt,
                        system_prompt=kks_system_prompt.replace("{num_player}", str(game.num_player)),
                        model=config.model,
                        reasoning_effort=config.reasoning_effort,
                        verbosity=config.verbosity
                    )
                elif config.provider == "gemini":
                    response_obj = client.chat_completion(
                        user_prompt=adjustment_prompt,
                        system_prompt=kks_system_prompt.replace("{num_player}", str(game.num_player)),
                        model=config.model,
                        response_schema=kks_response_schema
                    )
                else:
                    response_obj = client.chat_completion(
                        user_prompt=adjustment_prompt,
                        system_prompt=kks_system_prompt.replace("{num_player}", str(game.num_player)),
                        model=config.model
                    )
                
                # Parse response
                player_assignments, explanation = self._parse_agent_response(response_obj.text)
                
                return AgentResponse(
                    agent_name=agent_name,
                    game_id=game.game_id,
                    round_number=round_num,
                    phase="self_adjustment",
                    player_role_assignments=player_assignments,
                    explanation=explanation,
                    response_obj=response_obj,
                    timestamp=datetime.now().isoformat()
                )
                
            except Exception as e:
                print(f"Error in self-adjustment for {agent_name}: {e}")
                return AgentResponse(
                    agent_name=agent_name,
                    game_id=game.game_id,
                    round_number=round_num,
                    phase="self_adjustment",
                    player_role_assignments={},
                    explanation=f"Error: {str(e)}",
                    timestamp=datetime.now().isoformat()
                )
        
        # Get adjustment responses from all agents
        agent_names = [name for name in self.agents.keys()]
        responses = self.parallel_processor.process_tasks(
            get_adjustment_response, agent_names, preserve_order=True
        )
        
        return [r for r in responses if r is not None]
    
    def _conduct_final_vote(self, game: ground_truth, debate_rounds: List[DebateRound]) -> Dict[str, str]:
        """Conduct final majority vote for all players."""
        print("Conducting final majority vote...")
        
        # Get all player names
        player_names = list(debate_rounds[0].agent_responses[0].player_role_assignments.keys())
        final_vote = {}
        
        for player_name in player_names:
            # Collect votes for this player from the last round
            votes = []
            for round_data in debate_rounds:
                for response in round_data.agent_responses:
                    if player_name in response.player_role_assignments:
                        votes.append(response.player_role_assignments[player_name])
            
            # Find majority vote
            if votes:
                from collections import Counter
                vote_counts = Counter(votes)
                majority_role = vote_counts.most_common(1)[0][0]
                final_vote[player_name] = majority_role
                print(f"  {player_name}: {majority_role} (votes: {dict(vote_counts)})")
        
        return final_vote
    
    def _get_supervisor_decision(self, game: ground_truth, debate_rounds: List[DebateRound]) -> Dict[str, str]:
        """Get supervisor decision when consensus cannot be reached."""
        print("Getting supervisor decision...")
        
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
            supervisor_prompt = self._create_supervisor_prompt(game, debate_rounds)
            
            # Make API call
            if self.config.supervisor_provider == "openai":
                response_obj = supervisor_client.response_completion(
                    user_prompt=supervisor_prompt,
                    system_prompt=kks_system_prompt.replace("{num_player}", str(game.num_player)),
                    model=self.config.supervisor_model,
                    reasoning_effort="high",
                    verbosity="high"
                )
            elif self.config.supervisor_provider == "gemini":
                response_obj = supervisor_client.chat_completion(
                    user_prompt=supervisor_prompt,
                    system_prompt=kks_system_prompt.replace("{num_player}", str(game.num_player)),
                    model=self.config.supervisor_model,
                    response_schema=kks_response_schema
                )
            else:
                response_obj = supervisor_client.chat_completion(
                    user_prompt=supervisor_prompt,
                    system_prompt=kks_system_prompt.replace("{num_player}", str(game.num_player)),
                    model=self.config.supervisor_model
                )
            
            # Parse supervisor response
            player_assignments, explanation = self._parse_agent_response(response_obj.text)
            print(f"Supervisor decision: {player_assignments}")
            
            return player_assignments
            
        except Exception as e:
            print(f"Error getting supervisor decision: {e}")
            return {}
    
    def _parse_agent_response(self, response_text: str) -> Tuple[Dict[str, str], str]:
        """Parse agent response to extract player assignments and explanation."""
        try:
            # Try to parse JSON
            response_data = json.loads(response_text)
            
            # Extract player assignments
            player_assignments = {}
            explanation = ""
            
            if "players" in response_data and isinstance(response_data["players"], list):
                # New format with players array
                for player_data in response_data["players"]:
                    if isinstance(player_data, dict) and "name" in player_data and "role" in player_data:
                        player_assignments[player_data["name"]] = player_data["role"]
                explanation = response_data.get("explanation", "")
            else:
                # Old format - direct name: role mapping
                for key, value in response_data.items():
                    if key != "explanation" and isinstance(value, str):
                        player_assignments[key] = value
                explanation = response_data.get("explanation", "")
            
            return player_assignments, explanation
            
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract using regex
            import re
            pattern = r'(\w+)\s+is\s+a\s+(knight|knave|spy)\.'
            matches = re.findall(pattern, response_text, re.IGNORECASE)
            
            player_assignments = {}
            for player_name, role in matches:
                player_assignments[player_name] = role.lower()
            
            return player_assignments, response_text
    
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
            print(f"Error parsing ground truth: {e}")
            return {}
    
    def _create_debate_prompt(self, game: ground_truth, player_name: str,
                            initial_proposals: List[AgentResponse],
                            previous_rounds: List[DebateRound]) -> str:
        """Create debate prompt for a specific player's role."""
        
        prompt = f"""You are participating in a multi-agent debate about a Knight-Knaves-Spy game.

GAME INFORMATION:
{game.text_game}

CURRENT FOCUS: We are debating the role of {player_name}.

OTHER AGENTS' INITIAL PROPOSALS:
"""
        
        for proposal in initial_proposals:
            prompt += f"\n{proposal.agent_name} thinks {player_name} is a {proposal.player_role_assignments.get(player_name, 'unknown')}."
            prompt += f" Their reasoning: {proposal.explanation[:200]}...\n"
        
        if previous_rounds:
            prompt += "\nPREVIOUS DEBATE ROUNDS:\n"
            for round_data in previous_rounds:
                prompt += f"\nRound {round_data.round_number} - {round_data.player_name}:\n"
                for response in round_data.agent_responses:
                    if round_data.player_name in response.player_role_assignments:
                        prompt += f"  {response.agent_name}: {round_data.player_name} is a {response.player_role_assignments[round_data.player_name]}\n"
        
        prompt += f"""

Please provide your updated analysis focusing on {player_name}'s role. Consider the other agents' arguments and provide your reasoning. 

Return your response in the same JSON format as before:
{{
    "players": [
        {{"name": "player_name", "role": "role"}},
        ...
    ],
    "explanation": "Your detailed reasoning focusing on {player_name}'s role"
}}"""
        
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
            prompt += f"\n{response.agent_name} thinks {player_name} is a {response.player_role_assignments.get(player_name, 'unknown')}."
            prompt += f" Their reasoning: {response.explanation[:200]}...\n"
        
        prompt += f"""

Based on the debate, please provide your final assessment. You may adjust your position on {player_name} or any other players if you've been convinced by the arguments.

Return your response in the same JSON format:
{{
    "players": [
        {{"name": "player_name", "role": "role"}},
        ...
    ],
    "explanation": "Your final reasoning after considering the debate"
}}"""
        
        return prompt
    
    def _create_supervisor_prompt(self, game: ground_truth, debate_rounds: List[DebateRound]) -> str:
        """Create supervisor prompt for final decision."""
        
        prompt = f"""You are a supervisor AI tasked with making the final decision in a multi-agent debate about a Knight-Knaves-Spy game.

GAME INFORMATION:
{game.text_game}

DEBATE HISTORY:
"""
        
        for round_data in debate_rounds:
            prompt += f"\n--- Round {round_data.round_number}: {round_data.player_name} ---\n"
            for response in round_data.agent_responses:
                prompt += f"{response.agent_name}: {response.player_role_assignments}\n"
                prompt += f"Reasoning: {response.explanation[:300]}...\n"
        
        prompt += """

Based on the complete debate history, please provide your final decision. Consider all arguments and reasoning provided by the agents.

Return your response in the same JSON format:
{
    "players": [
        {"name": "player_name", "role": "role"},
        ...
    ],
    "explanation": "Your final decision with reasoning"
}"""
        
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
            "round_by_round": []
        }
        
        # Calculate initial accuracy for each agent
        for proposal in initial_proposals:
            correct = 0
            total = len(gt_solution)
            for player, role in gt_solution.items():
                if proposal.player_role_assignments.get(player) == role:
                    correct += 1
            tracking["initial_accuracy"][proposal.agent_name] = correct / total if total > 0 else 0
        
        # Calculate final accuracy
        if final_vote:
            correct = 0
            total = len(gt_solution)
            for player, role in gt_solution.items():
                if final_vote.get(player) == role:
                    correct += 1
            tracking["final_accuracy"]["majority_vote"] = correct / total if total > 0 else 0
        
        if supervisor_decision:
            correct = 0
            total = len(gt_solution)
            for player, role in gt_solution.items():
                if supervisor_decision.get(player) == role:
                    correct += 1
            tracking["final_accuracy"]["supervisor"] = correct / total if total > 0 else 0
        
        # Round-by-round tracking
        for round_data in debate_rounds:
            round_tracking = {
                "player": round_data.player_name,
                "round": round_data.round_number,
                "agent_accuracy": {}
            }
            
            for response in round_data.agent_responses:
                correct = 0
                total = len(gt_solution)
                for player, role in gt_solution.items():
                    if response.player_role_assignments.get(player) == role:
                        correct += 1
                round_tracking["agent_accuracy"][response.agent_name] = correct / total if total > 0 else 0
            
            tracking["round_by_round"].append(round_tracking)
        
        return tracking
    
    def _save_debate_session(self, session: DebateSession):
        """Save debate session to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"debate_session_{session.game_id}_{timestamp}.json"
        filepath = os.path.join(self.config.output_path, filename)
        
        # Convert to serializable format
        session_dict = asdict(session)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(session_dict, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Debate session saved to: {filepath}")
    
    def run_batch_debate(self, games: List[ground_truth]) -> List[DebateSession]:
        """Run debate sessions for multiple games."""
        sessions = []
        
        for i, game in enumerate(games):
            print(f"\n{'='*60}")
            print(f"Running debate session {i+1}/{len(games)}")
            print(f"{'='*60}")
            
            session = self.run_debate_session(game)
            sessions.append(session)
        
        return sessions
