"""
Enhanced prompts for multi-agent debate system.
"""

# Base system prompt for debate
debate_system_prompt = """
You are participating in a multi-agent debate about a Knight-Knaves-Spy game. You are one of several AI agents working together to solve the puzzle through collaborative reasoning.

GAME RULES:
- Knights always tell the truth
- Knaves always lie  
- Spies can either tell the truth or lie
- The hints from the game manager are always true

DEBATE CONTEXT:
You will see other agents' proposals and reasoning. Your task is to:
1. Consider other agents' arguments carefully
2. Provide your own analysis and reasoning
3. Be open to changing your position based on strong arguments
4. Focus on logical consistency and evidence
5. Maintain a collaborative and constructive tone

RESPONSE FORMAT:
Always respond in the following JSON format:
{
    "players": [
        {"name": "player_name", "role": "role"},
        ...
    ],
    "explanation": "Your detailed reasoning and analysis",
    "confidence": 0.0-1.0,
    "key_insights": ["insight1", "insight2", ...]
}

Be thorough in your reasoning and consider all available information.
"""

# Initial proposal prompt
initial_proposal_prompt = """
You are participating in a multi-agent debate about a Knight-Knaves-Spy game. This is your initial analysis.

GAME INFORMATION:
{game_text}

Please provide your initial solution to this puzzle. Consider all statements carefully and provide detailed reasoning for your conclusions.

Focus on:
1. Identifying which statements are true/false based on the rules
2. Determining each player's role through logical deduction
3. Explaining your reasoning step by step
4. Being confident but open to revision based on other agents' input

Return your response in the specified JSON format with your complete analysis.
"""

# Debate phase prompt
debate_phase_prompt = """
You are in the debate phase of a multi-agent discussion about a Knight-Knaves-Spy game.

GAME INFORMATION:
{game_text}

CURRENT FOCUS: We are specifically debating the role of {player_name}.

OTHER AGENTS' PROPOSALS:
{other_proposals}

PREVIOUS DEBATE ROUNDS:
{previous_rounds}

Your task:
1. Consider all other agents' arguments about {player_name}'s role
2. Evaluate the strength of their reasoning
3. Provide your own analysis focusing on {player_name}
4. Be open to changing your position if you find compelling arguments
5. Explain why you agree or disagree with other agents

Focus specifically on {player_name}'s role while considering the broader context of the game.

Return your response in the specified JSON format with your analysis.
"""

# Self-adjustment prompt
self_adjustment_prompt = """
You are in the self-adjustment phase after a debate round about a Knight-Knaves-Spy game.

GAME INFORMATION:
{game_text}

JUST COMPLETED DEBATE:
We just finished debating {player_name}'s role. Here's what happened:

DEBATE SUMMARY:
{debate_summary}

YOUR PREVIOUS POSITION:
{your_previous_position}

Your task:
1. Reflect on the debate that just occurred
2. Consider whether you should adjust your position on {player_name} or any other players
3. Provide your final assessment after considering all arguments
4. Be honest about what convinced you or what didn't
5. Update your complete solution if needed

You may:
- Stick to your original position if you're still convinced
- Change your mind about {player_name} if other arguments were stronger
- Adjust other players' roles if the debate revealed new insights
- Express uncertainty if the evidence is inconclusive

Return your response in the specified JSON format with your final assessment.
"""

# Supervisor decision prompt
supervisor_decision_prompt = """
You are a supervisor AI tasked with making the final decision in a multi-agent debate about a Knight-Knaves-Spy game.

GAME INFORMATION:
{game_text}

COMPLETE DEBATE HISTORY:
{debate_history}

CONSENSUS STATUS:
- Final Vote: {final_vote}
- Consensus Reached: {consensus_reached}

Your role:
1. Review the entire debate history carefully
2. Evaluate the quality of reasoning from all agents
3. Consider the evidence and logical consistency
4. Make the final decision when agents cannot reach consensus
5. Provide clear reasoning for your decision

You have access to the complete context and should make the most logical decision based on the evidence presented.

Return your response in the specified JSON format with your final decision and reasoning.
"""

# Majority vote prompt
majority_vote_prompt = """
You are conducting a final majority vote for a Knight-Knaves-Spy game.

GAME INFORMATION:
{game_text}

DEBATE RESULTS:
{debate_results}

Your task:
1. Collect all final positions from the debate
2. Determine the majority opinion for each player's role
3. Identify any players where no clear majority exists
4. Report the consensus results

For each player, determine:
- The most commonly proposed role
- Whether there's a clear majority (>50%)
- Any ties or close calls that need supervisor attention

Return your response in the specified JSON format with the majority vote results.
"""

# Enhanced response schema for debate
debate_response_schema = {
    'type': 'object',
    'description': "A complete solution to a Knights and Knaves puzzle with debate context.",
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
        'explanation': {
            'type': 'string',
            'description': 'A step-by-step logical reasoning for the solution.'
        },
        'confidence': {
            'type': 'number',
            'minimum': 0.0,
            'maximum': 1.0,
            'description': 'Confidence level in the solution (0.0 to 1.0)'
        },
        'key_insights': {
            'type': 'array',
            'items': {
                'type': 'string'
            },
            'description': 'Key insights or discoveries from the analysis'
        },
        'agreement_with': {
            'type': 'array',
            'items': {
                'type': 'string'
            },
            'description': 'Names of agents whose arguments you agree with'
        },
        'disagreement_with': {
            'type': 'array',
            'items': {
                'type': 'string'
            },
            'description': 'Names of agents whose arguments you disagree with'
        }
    },
    'required': ['players', 'explanation', 'confidence']
}

def create_debate_prompt(game_text: str, player_name: str, 
                        other_proposals: str, previous_rounds: str) -> str:
    """Create a debate prompt for a specific player's role."""
    return debate_phase_prompt.format(
        game_text=game_text,
        player_name=player_name,
        other_proposals=other_proposals,
        previous_rounds=previous_rounds
    )

def create_self_adjustment_prompt(game_text: str, player_name: str,
                                 debate_summary: str, your_previous_position: str) -> str:
    """Create a self-adjustment prompt."""
    return self_adjustment_prompt.format(
        game_text=game_text,
        player_name=player_name,
        debate_summary=debate_summary,
        your_previous_position=your_previous_position
    )

def create_supervisor_prompt(game_text: str, debate_history: str,
                           final_vote: str, consensus_reached: bool) -> str:
    """Create a supervisor decision prompt."""
    return supervisor_decision_prompt.format(
        game_text=game_text,
        debate_history=debate_history,
        final_vote=final_vote,
        consensus_reached=consensus_reached
    )

def create_initial_proposal_prompt(game_text: str) -> str:
    """Create an initial proposal prompt."""
    return initial_proposal_prompt.format(game_text=game_text)
