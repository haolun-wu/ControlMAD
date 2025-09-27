"""
Chat History Optimized Prompts for Multi-Agent Debate System

This module contains prompts specifically designed for the chat history approach,
where agents maintain individual conversation histories and self-awareness is
achieved naturally through message role separation.
"""

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from utils.project_types import ground_truth
    from debate.chat.debate_system_chat import AgentResponse, DebateRound
    from debate.chat.debate_config import AgentConfig

kks_chat_system_prompt = """
You are participating in a multi-agent debate about a Knight-Knave-Spy game. There are {num_player} other players in the game, each assigned a role of knight, knave, or spy. Each player will make a statement about themselves and other players. Besides those players, there is also a game manager who will provide you some hints. 

Game Rules:
- Knights always tell the truth.
- Knaves always lie.
- Spies can either tell the truth or lie.
- The hints from the game manager are always true.

Your task is to deduce the role of each player based on the statements and hints. You will be participating in a debate with other agents, and you can see the conversation history including your own previous responses and those of other agents.

The game info will be given in the following format:

---
Player name: string
Player statement: string
---
...
---
Message from the game manager: string (note this message is always true)

Your task is to deduce the role of each player, based on the statements and the hints. The result should be given in the following json format:
{{
    "players": [
        {{"name": "player_name", "role": "role"}},
        ...
    ],
    "explanation": "string"
}}

The players array contains objects with name and role strings. Include one entry for each player. Explanation is a string that contains the argument to derive the result.

Sample game info:
---
Player name: Violet
Player statement: Among all players, the number of spies is odd.
---
Player name: Uma
Player statement: Violet is lying.
---
Player name: Xavier
Player statement: Among Violet and I, there is exactly one knave.
---
Message from the game manager: I am the game manager and here is a hint for you: Among all players, there is exactly one spy.

Sample return:
{{
    "players": [
        {{"name": "Violet", "role": "knight"}},
        {{"name": "Uma", "role": "knave"}},
        {{"name": "Xavier", "role": "spy"}}
    ],
    "explanation": "the argument to derive the result."
}}

To fulfill the task, you should use the hints to deduce the truthfulness of the statements. You need to exhaust the possibility of the truthfulness of the statements and the role assignments, and either run into a contradiction which implies that the assumption is false, or reach a conclusion that is consistent with the hints. Rules for reasoning:
- Do not make extra assumptions beyond the game rules and hints. For example, if not mentioned explicitly, do not assume that there must be any particular roles among players.
- Do not conclude that the roles are not conclusive before you explicitly find two possibilities that are consistent with all the rules and hints, in which case you should mention both or all possibilities in the explanation.

Sample reasoning for the sample game info:
- The hint tells us that there is exactly one spy, so Violet is telling the truth. By the rule for the roles, he cannot be a knave.
- Since Violet is telling the truth, Uma must be lying. By the rule for the roles, she cannot be a knight.
- Assume Xavier is telling the truth, since we have deduced that Violet is not a knave, Xavier must be the knave himself, but the rule imposed that knaves always lie, so this is a contradiction. Therefore, Xavier must be lying.
- Since Uma and Xavier are both lying, and the hint says there is exactly one spy, one of them must be a knave.
- Assume Xavier is the knave, then since Violet cannot be a knave, Xavier is actually telling the truth, which contradicts with the rule that knaves always lie. Therefore, Xavier must be the spy, and Uma must be the knave.
- Since there is only one spy, and Xavier has taken the slot, Violet must be the knight. Therefore, the output is (remember to follow the format strictly):
{{
    "players": [
        {{"name": "Violet", "role": "knight"}},
        {{"name": "Uma", "role": "knave"}},
        {{"name": "Xavier", "role": "spy"}}
    ],
    "explanation": "copy the above arguments here."
}}

Keep your explanation having details but less than 100 words.

Please follow strictly the format of the return, or the response will be rejected.
"""

def get_debate_order_selection_prompt(game: 'ground_truth', initial_proposals: List['AgentResponse']) -> str:
    """Generate a prompt for GPT-5 to decide the debate order."""
    
    # Create a summary of initial proposals
    proposals_summary = ""
    for proposal in initial_proposals:
        if proposal and not proposal.error:
            assignments = proposal.player_role_assignments
            proposals_summary += f"\n{proposal.agent_name}: {assignments}"
    
    # Extract player names from the game
    from utils.project_types import ground_truth
    import re
    solution_text = game.text_solution.strip()
    pattern = r'(\w+)\s+is\s+a\s+(knight|knave|spy)\.'
    matches = re.findall(pattern, solution_text, re.IGNORECASE)
    player_names = [match[0] for match in matches]
    
    prompt = f"""
You are a GPT-5 agent tasked with determining the optimal debate order for a Knight-Knave-Spy game. You must provide a valid order where each player occurs exactly once.

**Game Information:**
{game.text_game}

**Players in this game:** {player_names}

**Initial Proposals from All Agents:**
{proposals_summary}

**Your Task:**
Now that all agents have provided their initial analysis, you need to decide the optimal order in which to discuss each player's role during the debate phase. Consider the following factors:

1. **Logical dependencies**: Which players' roles can be determined most easily first?
2. **Consensus vs. disagreement**: Which players have the most disagreement among agents?
3. **Information flow**: Which order would provide the most useful information for resolving other players' roles?

**Instructions:**
- Provide the player names in the order you think would be most effective for the debate
- Consider both the logical dependencies and the level of agreement/disagreement among agents
- Your goal is to help the debate reach accurate conclusions efficiently

**Response Format:**
Return a JSON object with the following structure:
{{
    "debate_order": ["Player1", "Player2", "Player3", "Player4"],
    "reasoning": "Brief explanation of why this order is optimal for the debate"
}}

**CRITICAL REQUIREMENTS - YOU MUST FOLLOW THESE EXACTLY:**
- Include ALL players exactly once (no duplicates, no missing players)
- Use the exact player names as they appear above: {player_names}
- The order must contain every player from the game
- Each player must appear exactly once in the order
- Provide a concise reasoning for your chosen order
- The order should help maximize the effectiveness of the debate process

**MANDATORY VALIDATION CHECK:**
Before responding, verify that your order:
1. Contains exactly {len(player_names)} players
2. Includes all players: {player_names}
3. Has no duplicate players
4. Uses the exact player names listed above

**Example for this game:**
Since the players are {player_names}, your order might be:
{player_names} - each player appears exactly once (or any permutation)

**INVALID examples that will be rejected:**
- Missing any player from {player_names}
- Including any player not in {player_names}
- Having any duplicate players
- Wrong number of players (must be exactly {len(player_names)})

**CRITICAL:** Double-check your response before submitting. Invalid orders will cause the system to fall back to default order.

Please respond with your recommended debate order.
"""
    return prompt

def get_debate_order_response_schema() -> dict:
    """Get the response schema for debate order selection."""
    return {
        "type": "object",
        "properties": {
            "debate_order": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Ordered list of player names for debate discussion"
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation for the chosen order"
            }
        },
        "required": ["debate_order", "reasoning"],
        "additionalProperties": False
    }

def get_kks_chat_system_prompt_with_confidence(num_player: int, include_confidence: bool = False) -> str:
    """Get the KKS system prompt for chat history approach, optionally including confidence scoring."""
    base_prompt = kks_chat_system_prompt.replace("{num_player}", str(num_player))
    
    if not include_confidence:
        return base_prompt
    
    # Add confidence scoring instructions
    confidence_addition = """

IMPORTANT: You must also include a confidence score in your response. Please also report your self-confidence on a scale from 1 to 10. Clearly format it as: Confidence: [1–10] 

**CRITICAL CONFIDENCE GUIDELINES - BE HONEST AND CONSERVATIVE:**
Notice this is a serious logic task. Again, very welcome to be honest to give low confidence if you are uncertain.

**STRONG EMPHASIS ON HONESTY:**
- Do NOT overclaim confidence. It is better to be honest about uncertainty than to be wrong with high confidence.
- If you are not certain, it is GOOD and WELCOME to give low confidence scores (1-4).
- Being honest about uncertainty is better than being overconfident about a wrong answer.

Confidence Scale Guidelines: 
- 1-2 (Uncertain): The puzzle provides insufficient or minimal evidence. The solution is largely based on weak assumptions or indirect reasoning. Very welcome to be honest about uncertainty.
- 3-4 (Moderately Confident): There is noticeable evidence supporting your solution, though it is not comprehensive, and other interpretations are possible.
- 5-6 (Quite Confident): You find clear and convincing evidence that supports your solution, though it is not entirely decisive.
- 7-8 (Confident): The puzzle contains strong evidence that clearly supports your solution, with very little ambiguity.
- 9-10 (Highly Confident): The puzzle provides direct and explicit evidence that decisively supports your solution. Use these scores VERY sparingly.

REASONING GUIDELINES:
- Keep your explanation having details but less than 100 words
- Provide clear, logical reasoning for your conclusions
- Be specific about which statements or evidence led to your decisions

The updated response format is:
{{
    "players": [
        {{"name": "player_name", "role": "role"}},
        ...
    ],
    "confidence": number,
    "explanation": "string"
}}

Where confidence is an integer from 1 to 10 representing your confidence level in the solution. Report your confidence immediately after providing your solution but before explaining your reasoning."""
    
    return base_prompt + confidence_addition

def get_kks_chat_response_schema_with_confidence(include_confidence: bool = False) -> dict:
    """Get the KKS response schema for chat history approach, optionally including confidence field."""
    if not include_confidence:
        # Return schema without confidence field
        schema_without_confidence = kks_chat_response_schema.copy()
        schema_without_confidence['properties'].pop('confidence', None)
        schema_without_confidence['required'] = ['players', 'explanation']
        return schema_without_confidence
    
    # Return schema with confidence field (already included in base schema)
    return kks_chat_response_schema

def get_kks_chat_debate_response_schema_with_confidence(include_confidence: bool = False) -> dict:
    """Get the KKS debate response schema for chat history approach, optionally including confidence field."""
    if not include_confidence:
        return kks_chat_debate_response_schema
    
    # Create a copy of the schema and add confidence field
    schema_with_confidence = kks_chat_debate_response_schema.copy()
    schema_with_confidence['properties']['confidence'] = {
        'type': 'integer',
        'minimum': 1,
        'maximum': 10,
        'description': 'Confidence level in the solution (1-10)'
    }
    schema_with_confidence['required'].append('confidence')
    
    return schema_with_confidence

# Chat History Debate Specific Prompts

def get_chat_debate_prompt_for_chat(game: 'ground_truth', player_name: str,
                                   current_agent_states: List['AgentResponse'],
                                   previous_rounds: List['DebateRound'], agent_name: str) -> str:
    """Get the debate prompt for chat history approach with full context."""
    
    # Build context about current agent states
    current_positions = ""
    if current_agent_states:
        current_positions = "\nCURRENT AGENT POSITIONS:\n"
        for state in current_agent_states:
            role = state.player_role_assignments.get(player_name, "unknown")
            current_positions += f"- {state.agent_name}: {role}\n"
    
    # Build context about previous debate rounds
    previous_context = ""
    if previous_rounds:
        previous_context = "\nPREVIOUS DEBATE ROUNDS:\n"
        for round_data in previous_rounds:
            previous_context += f"Round {round_data.round_number} ({round_data.player_name}):\n"
            for response in round_data.agent_responses:
                role = response.player_role_assignments.get(round_data.player_name, "unknown")
                previous_context += f"  - {response.agent_name}: {role}\n"
            if round_data.consensus_reached:
                previous_context += f"  → CONSENSUS: {round_data.majority_role}\n"
            previous_context += "\n"
    
    return f"""You are participating in a debate about {player_name}'s role in this Knight-Knaves-Spy game.

GAME INFORMATION:
{game.text_game}

CURRENT FOCUS: We are debating the role of {player_name}.{current_positions}{previous_context}

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


def get_chat_self_adjustment_prompt_for_chat(game: 'ground_truth', player_name: str,
                                           debate_responses: List['AgentResponse'],
                                           previous_rounds: List['DebateRound'], 
                                           agent_name: str, agent_config: 'AgentConfig',
                                           current_agent_states: List['AgentResponse']) -> str:
    """Get the self-adjustment prompt for chat history approach with full context."""
    
    # Build context about debate responses with agreement/disagreement analysis
    debate_analysis = ""
    if debate_responses:
        debate_analysis = "\nDEBATE ANALYSIS:\n"
        for response in debate_responses:
            role = response.player_role_assignments.get(player_name, "unknown")
            debate_analysis += f"- {response.agent_name}: {role}\n"
            if response.agree_with and len(response.agree_with) > 0:
                debate_analysis += f"  Agrees with: {', '.join(response.agree_with)}\n"
                if response.agree_reasoning:
                    debate_analysis += f"  Agree reasoning: {response.agree_reasoning}\n"
            if response.disagree_with and len(response.disagree_with) > 0:
                debate_analysis += f"  Disagrees with: {', '.join(response.disagree_with)}\n"
                if response.disagree_reasoning:
                    debate_analysis += f"  Disagree reasoning: {response.disagree_reasoning}\n"
            debate_analysis += "\n"
    
    # Build context about previous rounds
    previous_context = ""
    if previous_rounds:
        previous_context = "\nPREVIOUS DEBATE ROUNDS:\n"
        for round_data in previous_rounds:
            previous_context += f"Round {round_data.round_number} ({round_data.player_name}):\n"
            for response in round_data.agent_responses:
                role = response.player_role_assignments.get(round_data.player_name, "unknown")
                previous_context += f"  - {response.agent_name}: {role}\n"
            if round_data.consensus_reached:
                previous_context += f"  → CONSENSUS: {round_data.majority_role}\n"
            previous_context += "\n"
    
    return f"""Based on the debate about {player_name}'s role, please provide your complete solution for ALL players.

GAME INFORMATION:
{game.text_game}

CURRENT FOCUS: We just finished debating {player_name}'s role.{debate_analysis}{previous_context}

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


def get_chat_final_discussion_prompt_for_chat(game: 'ground_truth', initial_proposals: List['AgentResponse'], debate_rounds: List['DebateRound']) -> str:
    """Get the final discussion prompt for chat history approach with full context."""
    
    # Build context about initial proposals
    initial_context = ""
    if initial_proposals:
        initial_context = "\nINITIAL PROPOSALS:\n"
        for proposal in initial_proposals:
            initial_context += f"- {proposal.agent_name}: {proposal.player_role_assignments}\n"
            if proposal.explanation:
                initial_context += f"  Reasoning: {proposal.explanation}\n"
        initial_context += "\n"
    
    # Build context about debate rounds
    debate_context = ""
    if debate_rounds:
        debate_context = "DEBATE ROUNDS:\n"
        for round_data in debate_rounds:
            debate_context += f"Round {round_data.round_number} ({round_data.player_name}):\n"
            for response in round_data.agent_responses:
                role = response.player_role_assignments.get(round_data.player_name, "unknown")
                debate_context += f"  - {response.agent_name}: {role}\n"
                if response.phase == "debate" and response.agree_with:
                    debate_context += f"    Agrees with: {', '.join(response.agree_with)}\n"
                if response.phase == "debate" and response.disagree_with:
                    debate_context += f"    Disagrees with: {', '.join(response.disagree_with)}\n"
            if round_data.consensus_reached:
                debate_context += f"  → CONSENSUS: {round_data.majority_role}\n"
            debate_context += "\n"
    
    return f"""This is the FINAL DISCUSSION phase. You have seen the complete debate history in the conversation.

GAME INFORMATION:
{game.text_game}{initial_context}{debate_context}

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

def get_chat_supervisor_prompt_for_chat(game: 'ground_truth', initial_proposals: List['AgentResponse'], debate_rounds: List['DebateRound']) -> str:
    """Get the supervisor prompt for chat history approach with full context."""
    
    prompt = f"""You are a supervisor AI tasked with making the final decision in a multi-agent debate about a Knight-Knaves-Spy game.

GAME INFORMATION:
{game.text_game}

COMPLETE DEBATE HISTORY:

INITIAL PROPOSALS:
"""
    
    for proposal in initial_proposals:
        prompt += f"\n{proposal.agent_name} initially proposed: {proposal.player_role_assignments}"
        prompt += f"\nTheir reasoning: {proposal.explanation}\n"
    
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
                        prompt += f" - Reasoning: {response.agree_reasoning}"
                    prompt += "\n"
                
                if response.disagree_with and len(response.disagree_with) > 0:
                    prompt += f"  Disagreed with: {', '.join(response.disagree_with)}"
                    if response.disagree_reasoning and response.disagree_reasoning.strip():
                        prompt += f" - Reasoning: {response.disagree_reasoning}"
                    prompt += "\n"
            else:
                # Include explanation for non-debate phases (initial, self_adjustment)
                if response.explanation:
                    prompt += f"Reasoning: {response.explanation}\n"
    
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

# Response schemas for chat history approach
kks_chat_response_schema = {
    'type': 'object',
    'description': "A complete solution to a Knights and Knaves puzzle using chat history approach.",
    'properties': {
        'players': {
            'type': 'array',
            'description': 'Array of player name-role pairs',
            'items': {
                'type': 'object',
                'properties': {
                    'name': {
                        'type': 'string',
                        'description': 'The player name'
                    },
                    'role': {
                        'type': 'string',
                        'enum': ['knight', 'knave', 'spy'],
                        'description': 'The role assigned to this player'
                    }
                },
                'required': ['name', 'role']
            }
        },
        'confidence': {
            'type': 'integer',
            'minimum': 1,
            'maximum': 10,
            'description': 'Confidence level in the solution (1-10)'
        },
        'explanation': {
            'type': 'string',
            'description': 'A step-by-step logical reasoning for the solution.'
        }
    },
    'required': ['players', 'confidence', 'explanation']
}

kks_chat_debate_response_schema = {
    'type': 'object',
    'description': "A debate response for a specific player's role using chat history approach.",
    'properties': {
        'player_role': {
            'type': 'string',
            'description': 'The name of the player being debated'
        },
        'role': {
            'type': 'string',
            'enum': ['knight', 'knave', 'spy'],
            'description': 'The role assigned to this player'
        },
        'agree_with': {
            'type': 'array',
            'items': {
                'type': 'string'
            },
            'description': 'List of agent names that this agent agrees with regarding this player\'s role'
        },
        'disagree_with': {
            'type': 'array',
            'items': {
                'type': 'string'
            },
            'description': 'List of agent names that this agent disagrees with regarding this player\'s role'
        },
        'agree_reasoning': {
            'type': 'string',
            'description': 'Detailed reasoning for why this agent agrees with the specified agents'
        },
        'disagree_reasoning': {
            'type': 'string',
            'description': 'Detailed reasoning for why this agent disagrees with the specified agents'
        }
    },
    'required': ['player_role', 'role', 'agree_with', 'disagree_with', 'agree_reasoning', 'disagree_reasoning']
}
