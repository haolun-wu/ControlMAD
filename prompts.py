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

Be concise with the explanation.

Please follows strictly the format of the return, or the response will be rejected.
"""

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
        'explanation': {
            'type': 'string',
            'description': 'A step-by-step logical reasoning for the solution.'
        }
    },
    'required': ['players', 'explanation']
}