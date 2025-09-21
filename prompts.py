kks_system_prompt = """
You are playing a Knight-Knave-Spy game. There are {num_player} other players in the game, each of them is assigned a role of knight, knave, or spy. Each player will make a statement about themselves and other players. Besides those players, there is also a game manager who will provide you some hints. Rule of the game:

- Knights always tell the truth.
- Knaves always lie.
- Spies can either tell the truth or lie.
- The hints from the game manager are always true.

The game info will be given in the following format:

---
Player name: string
Player statement: string
---
...
---
Message from the game manager: string (note this message is always true)

Your task is to deduce the role of each player, based on the statements and the hints. The result should be given in the following json format:
{
    "players": [
        {"name": "player_name", "role": "role"},
        ...
    ],
    "explanation": "string"
}
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
{
    "players": [
        {"name": "Violet", "role": "knight"},
        {"name": "Uma", "role": "knave"},
        {"name": "Xavier", "role": "spy"}
    ],
    "explanation": "the argument to derive the result."
}

To fulfill the task, you should use the hints to deduce the truthfulness of the statements. You need to exhaust the possibility of the truthfulness of the statements and the role assignments, and either run into a contradiction which implies that the assumption is false, or reach a conclusion that is consistent with the hints. Rules for reasoning:
- Do not make extra assumptions beyond the game rules and hints. for example, if not mentioned explicitly, do not assume that there must be any particular roles among players.
- Do not conclude that the roles are not conclusive before you explicitly find two possibilities that are consistent with all the rules and hints, in which case you should mention both or all possibilities in the explanation.

Sample reasoning for the sample game info:
- The hint tells us that there is exactly one spy, so Violet is telling the truth. By the rule for the roles, he cannot be a knave.
- Since Violet is telling the truth, Uma must be lying. By the rule for the roles, she cannot be a knight.
- Assume Xavier is telling the truth, since we have deduced that Violet is not a knave, Xavier must be the knave himself, but the rule imposed that knaves always lie, so this is a contradiction. Therefore, Xavier must be lying.
- Since Uma and Xavier are both lying, and the hint says there is exactly one spy, one of them must be a knave.
- Assume Xavier is the knave, then since Violet cannot be a knave, Xavier is actually telling the truth, which contradicts with the rule that knaves always lie. Therefore, Xavier must be the spy, and Uma must be the knave.
- Since there is only one spy, and Xavier has taken the slot, Violet must be the knight. Therefore, the output is (remember to follow the format strictly):
{
    "players": [
        {"name": "Violet", "role": "knight"},
        {"name": "Uma", "role": "knave"},
        {"name": "Xavier", "role": "spy"}
    ],
    "explanation": "copy the above arguments here."
}

Keep your explanation having details but less than 50 words

Please follows strictly the format of the return, or the response will be rejected.
"""

def get_kks_system_prompt_with_confidence(num_player: int, include_confidence: bool = False) -> str:
    """Get the KKS system prompt, optionally including confidence scoring."""
    base_prompt = kks_system_prompt.replace("{num_player}", str(num_player))
    
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
- Keep your explanation having details but less than 50 words
- Provide clear, logical reasoning for your conclusions
- Be specific about which statements or evidence led to your decisions

The updated response format is:
{
    "players": [
        {"name": "player_name", "role": "role"},
        ...
    ],
    "confidence": number,
    "explanation": "string"
}

Where confidence is an integer from 1 to 10 representing your confidence level in the solution. Report your confidence immediately after providing your solution but before explaining your reasoning."""
    
    return base_prompt + confidence_addition

def get_kks_response_schema_with_confidence(include_confidence: bool = False) -> dict:
    """Get the KKS response schema, optionally including confidence field."""
    if not include_confidence:
        # Return schema without confidence field
        schema_without_confidence = kks_response_schema.copy()
        schema_without_confidence['properties'].pop('confidence', None)
        schema_without_confidence['required'] = ['players', 'explanation']
        return schema_without_confidence
    
    # Return schema with confidence field (already included in base schema)
    return kks_response_schema

def get_kks_debate_response_schema_with_confidence(include_confidence: bool = False) -> dict:
    """Get the KKS debate response schema, optionally including confidence field."""
    if not include_confidence:
        return kks_debate_response_schema
    
    # Create a copy of the schema and add confidence field
    schema_with_confidence = kks_debate_response_schema.copy()
    schema_with_confidence['properties']['confidence'] = {
        'type': 'integer',
        'minimum': 1,
        'maximum': 10,
        'description': 'Confidence level in the solution (1-10)'
    }
    schema_with_confidence['required'].append('confidence')
    
    return schema_with_confidence

kks_response_schema = {
    'type': 'object',
    'description': "A complete solution to a Knights and Knaves puzzle.",
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

kks_debate_response_schema = {
    'type': 'object',
    'description': "A debate response for a specific player's role in a Knights and Knaves puzzle.",
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