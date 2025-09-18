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

Your task is to deduce the role of each player, based on the statements and the hints. Please reply with the role of each player, your confidence score, and then your reasoning process. 

**IMPORTANT:** You must provide your confidence score immediately after stating your solution, BEFORE explaining your reasoning. This should be a number from 1 to 10 representing how confident you are in your solution (1 = very uncertain, 10 = completely certain).

**CRITICAL CONFIDENCE GUIDELINES - BE HONEST AND CONSERVATIVE:**
Notice this is a serious logic task. Again, very welcome to be honest to give low confidence if you are uncertain.

- These puzzles are designed to be tricky and subtle logical errors are very common.
- Most solutions that seem correct are actually wrong due to hidden assumptions or logical flaws.
- BE VERY CAUTIOUS about giving high confidence scores (8-10). Only use these if you are absolutely certain.
- It is GOOD and WELCOME to give low confidence scores (1-4) if you have any doubts.
- Use medium confidence (5-7) only if you believe your solution is likely correct but acknowledge potential issues.
- Remember: Being honest about uncertainty is better than being overconfident about a wrong answer.
- If you had to make any assumptions or if multiple interpretations seemed possible, use low confidence (1-4).
- IMPORTANT: Do not overclaim confidence. It is better to be honest about uncertainty than to be wrong with high confidence. 

<rules>
- Please include clear logical guides in your reasoning, including a clear indication of the beginning and ending of each logic loop. For example: "I will study three cases", "This ends the case 1", "Let us proof by contradiction", "Hence the assumption ... is false and we conclude ..."
- Do not make extra assumptions beyond the game rules and hints. for example, if not mentioned explicitly, do not assume that there must be any particular roles among players.
- The reasoning is the process you find the solution, not the explanation of the solution. In the reasoning, you are supposed to deduce the roles of all players from the given information. You **cannot** simply assume any player to have a particular role without any justification.
- If you think the time or token limit is too short, you can return a partial solution with a partial reasoning, e.g., cases you have tried, roles that can be determined.
- Precise and logical soundness is more important than completeness for an answer, as there will be following up rounds to fully develop the solution.
- There is a unique solution to the game, so if you find two or more possible solutions, you should have a double check what you have missed.
- Please provides a **detailed** reasoning.
</rules>

<example>
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

Sample reasoning for the sample game info:
- The hint tells us that there is exactly one spy, so Violet is telling the truth. By the rule for the roles, he cannot be a knave.
- Since Violet is telling the truth, Uma must be lying. By the rule for the roles, she cannot be a knight.
- Assume Xavier is telling the truth, since we have deduced that Violet is not a knave, Xavier must be the knave himself, but the rule imposed that knaves always lie, so this is a contradiction. Therefore, Xavier must be lying.
- Since Uma and Xavier are both lying, and the hint says there is exactly one spy, one of them must be a knave.
- Assume Xavier is the knave, then since Violet cannot be a knave, Xavier is actually telling the truth, which contradicts with the rule that knaves always lie. Therefore, Xavier must be the spy, and Uma must be the knave.
- Since there is only one spy, and Xavier has taken the slot, Violet must be the knight. This complete the reasoning.

Sample return:
{
    "players": [
        {"name": "Xavier", "role": "spy"},
        {"name": "Uma", "role": "knave"},
        {"name": "Violet", "role": "knight"}
    ],
    "confidence": 3,
    "reasoning": "copy the above arguments here"
}
"""

kks_response_schema_gemini = {
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
            'description': 'Confidence score from 1 to 10 representing how confident the model is in the solution.'
        },
        'reasoning': {
            'type': 'string',
            'description': 'A step-by-step logical reasoning for the solution.'
        }
    },
    'required': ['players', 'confidence', 'reasoning']
}

kks_response_schema_openai = {
    "type": "json_schema",
    "json_schema": {
        "name": "knights_and_knaves_solution",
        "strict": True,
        "schema": {
            "type": "object",
            "description": "A complete solution to a Knights and Knaves puzzle.",
            "properties": {
                "players": {
                    "type": "array",
                    "description": "Array of player name-role pairs",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The player name"
                            },
                            "role": {
                                "type": "string",
                                "enum": ["knight", "knave", "spy"],
                                "description": "The role assigned to this player"
                            }
                        },
                        "required": ["name", "role"],
                        "additionalProperties": False
                    }
                },
                "confidence": {
                    "type": "integer",
                    "description": "Confidence score from 1 to 10 representing how confident the model is in the solution."
                },
                "reasoning": {
                    "type": "string",
                    "description": "A step-by-step logical reasoning for the solution."
                }
            },
            "required": ["players", "confidence", "reasoning"],
            "additionalProperties": False
        }
    }
}

initial_parser_response_schema_gemini = {
    'type': 'array',
    'description': 'Logical order of role determinations in the reasoning',
    'items': {
        'type': 'string',
        'description': 'A role determination statement in the format "Name is a Role"'
    }
}

initial_parser_response_schema_openai = {
    "type": "json_schema",
    "json_schema": {
        "name": "parser_logical_order",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "logical_order": {
                    "type": "array",
                    "description": "Logical order of role determinations in the reasoning",
                    "items": {
                        "type": "string",
                        "description": "A role determination statement in the format 'Name is a Role'"
                    }
                }
            },
            "required": ["logical_order"],
            "additionalProperties": False
        }
    }
}

initial_parser_system_prompt = """
You are a parser for a Knight-Knave-Spy game. Your task is to parse a candidate solution of the game to extract the logic chain of the reasoning. This would help other agent evaluate the logical soundness of the solution. In the user's prompt, you will be given the original game instance and the candidate solution of the game, proposed by participants, of the following format:

# Game instance
(game_instance)

# Game solution
(game_solution of the form: "Name is a Role.")

# Game reasoning
(reasoning of the argument of the candidate solution)

Your goal is to extract the **logical order** of how the roles are deduced. Please reply in the following json format:
{
  "logical_order": ["Name_1 is a Role_1", "Name_2 is a Role_2", "Name_3 is a Role_3", ...]
}
The list order indicates the logical order of the arguments. Please note: 
- **Do not** attempt to solve the game or verify the solution: your task is sololy to determine the logical order of the arguments. 
- **Do not** directly use the order in the candidate solution as the logical order of the arguments. The order of the solution is in the order of the game instance, not the order of how they are deduced.
- The statement about role determinations may not go exactly of the form "Name is a Role.", but you should be able to infer the logical order of the arguments from the reasoning.
- If the reasoning does not provide a valid reasoning of the solution, or only provides an reasonings for a few players but not all, you can simply return a short or even empty list.
- If multiple solution possibilities are provided in the reasoning, please stick to the situation that the solution is concluded to extract the logical order of the arguments, and ignore the other possibilities.

# Sample Game Instance 1:
---
Player name: Zane
Player statement: Charlie and Uma are both knights.
---
Player name: Charlie
Player statement: Zane and I have the same role.
---
Player name: Uma
Player statement: Zane, Rachel, and I all have the same role.
---
Player name: Rachel
Player statement: Among the following two statements, exactly one is true: 
 (1). Among Uma and I, there is exactly one knave.
 (2). Among all players, the number of knaves is even.
---
Message from the game manager: I am the game manager and here is a hint for you: Among all players, there is exactly one spy.

# Sample Solution 1:
Zane is a spy.
Charlie is a knave.
Uma is a knave.
Rachel is a knight.

# Sample Reasoning 1:
Step 1: Use the manager's hint: exactly one spy among the four players. That means three players are either knights or knaves. We test cases by assuming who the spy could be and check consistency with all statements.

Case A: Zane is the spy.
- Zane's statement can be either true or false (spies aren’t constrained). Thus we only need the other statements to be consistent.
- Charlie cannot be a knight, because if Charlie were a knight, his statement ‘Zane and I have the same role’ would require Zane to be a knight as well, which contradicts Zane being the spy. So Charlie must be knave.
- Uma cannot be a knight either, because Uma’s statement would require Zane, Rachel, and Uma to all share the same role, which is impossible since Zane is the spy and Uma cannot be spy. Therefore Uma is knave.- Rachel must be a knight. With Zane spy, Charlie knave, Uma knave, Rachel knight, Rachel’s statement evaluates as true: (1) among Uma and Rachel, exactly one knave — true (Uma is knave, Rachel is not); (2) the total number of knaves is even — true (two knaves: Charlie and Uma).
This assignment is consistent with every rule and with the manager’s hint. So Case A is valid: Zane=spy, Charlie=knave, Uma=knave, Rachel=knight.

Case B: Charlie is the spy.
- Then Zane cannot be a knight, because Charlie being spy makes Zane’s statement about Charlie and Uma being knights false if Zane were knight. The only self-consistent option is Zane being knave.
- Uma cannot be a knight either, because her statement would claim Zane, Rachel, and Uma all share the same role; with Zane knave that would require Uma and Rachel to be knaves as well, which would violate consistency with Zane’s knave status and Rachel’s possible truthfulness. The viable path is Uma being knave.
- Rachel must be a knight. With Zane knave, Uma knave, Rachel knight, Rachel’s two-part statement evaluates as true: (1) among Uma and I, exactly one knave — true (Uma is knave, Rachel is not); (2) the total number of knaves is even — true (two knaves: Zane and Uma).
Thus Case B is also consistent: Zane=knave, Charlie=spy, Uma=knave, Rachel=knight.

Conclusion: The information given is compatible with at least two distinct solutions, both featuring Rachel as a knight and exactly one spy, with Uma as a knave and the remaining roles distributed between Zane and Charlie as spy/knave in the two valid configurations described above. Hence there are two consistent solutions:
- Solution 1: Zane=spy, Charlie=knave, Uma=knave, Rachel=knight.
- Solution 2: Zane=knave, Charlie=spy, Uma=knave, Rachel=knight.
In both solutions, Uma is a knave and Rachel is a knight, and there is exactly one spy among the four players.

# Sample Return 1:
{
  "logical_order": ["Zane is a spy", "Charlie is a knave", "Uma is a knave", "Rachel is a knight"]
}

<explanation>We stick to case A as it is consistent with the solution. Case B is ignored. The return then extracts the logical order of the arguments from the reasoning.</explanation>

# Sample Game Instance 2:
---
Player name: Tina
Player statement: Uma and I have the same role.
---
Player name: Xavier
Player statement: Tina and I have the same role.
---
Player name: Uma
Player statement: Among Xavier, Tina, and Zane, the number of knaves is even.
---
Player name: Zane
Player statement: Among Uma, Tina, and Xavier, exactly two people are lying.
---
Message from the game manager: I am the game manager and here is a hint for you: Among all players, there is exactly one spy.

# Sample Solution 2:
Tina is a knight. 
Xavier is a spy
Uma is a knave
Zane is a knave

# Sample Reasoning 2:
I attempted to assign exactly one spy among the four players (as the manager states). I explored the three plausible primary cases for where the spy sits (Tina, Xavier, or Uma/Zane). In each case I propagated the truth-values of all four statements according to Knight/Knave rules and kept the spy unconstrained (since spies may lie or tell the truth). After a careful case analysis, every branch led to a contradiction:
- If Tina were the spy, Xavier would have to be a knave, but then the derived truth-values for Uma’s and Zane’s statements force inconsistent roles for Uma and Zane (one of them must be a Knight while the content of their statements requires them to be Knaves, or vice versa).
- If Xavier were the spy, the remaining three players would have to be Knights/Knaves in a way that makes Tina’s and Uma’s statements mutually inconsistent with Zane’s claim about the number of liars among a subset.
- If Uma or Zane were the spy, similar tangled dependencies arise that force impossible combinations (e.g., forcing Tina to be both Knight and Knave, or forcing Zane’s lie-truth condition to be simultaneously true and false).
Because in every exhaustive subcase I reach a contradiction, there is no assignment of Knight/Knave/Spy roles to Tina, Xavier, Uma, and Zane that satisfies all four statements while also honoring the manager’s hint that there is exactly one spy among all players.
Conclusion: With the given statements and the manager’s hint, there is no consistent solution. If you’d like, I can re-check with a slightly different interpretation of “lying” (e.g., if spies must sometimes lie but not necessarily in the same way as knaves) or you could provide an alternative hint to resolve the ambiguity.

# Sample Return 2:
{
  "logical_order": []
}

<explanation>The answer does not explain how the solution is concluded, but only provides a few failed tries. So no logical order can be inferred.</explanation>

# Sample Game Instance 3:
---
Player name: Uma
Player statement: Among all players, the number of knights is odd.
---
Player name: Eve
Player statement: Among the following two statements, exactly one is true: 
 (1). Uma is telling the truth.
 (2). Among all players, the number of knaves is odd.
---
Player name: Ivy
Player statement: I am a knave.
---
Player name: Wendy
Player statement: Eve and Uma are either both telling truth or both lying.
---
Message from the game manager: I am the game manager and here is a hint for you: Among all players, there is exactly one spy.

# Sample Solution 3:
Uma is a knight.
Eve is a knight.
Ivy is a spy.
Wendy is a knight.

# Sample Reasoning 3:

First, Ivy’s statement is 'I am a knave.' This cannot be true for a knight or knave (knights always tell truth, knaves always lie), so Ivy cannot be knight or knave. The only compatible role for Ivy is spy. Since the manager’s hint says there is exactly one spy, Ivy is the sole spy. This fixes the remaining three players (Uma, Eve, Wendy) as knights or knaves, with three players total and one spy already used, so they must all be knights or knaves with no spies among them.

# Sample Return 3:
{
  "logical_order": ["Ivy is a spy"]
}

<explanation>Only Ivy's role is determined, so the return is a short list.</explanation>

# Sample Game Instance 4:
Player name: Yara
Player statement: Quinn and I have different roles.
---
Player name: Liam
Player statement: Quinn and I have different roles.
---
Player name: Violet
Player statement: Among Liam and Yara, exactly one person is lying.
---
Player name: Quinn
Player statement: Violet, Yara, and I do not all have the same role.
---
Message from the game manager: I am the game manager and here is a hint for you: Among all players, there is exactly one spy.

# Sample Solution 4:
Yara is a knight.
Liam is a knight.
Violet is a knave.
Quinn is a spy.

# Sample Reasoning 4:
Assume Quinn is the one spy (since the manager says exactly one spy exists). Then Yara’s and Liam’s statements about Quinn differ from themselves. If Quinn is the spy, Yara cannot be knave (a knave would have to lie about Quinn and herself having different roles, which would imply Yara is also the spy, impossible with exactly one spy). Therefore Yara must be a knight, and she is telling the truth that Quinn and she have different roles. For Liam, if he were knave, his statement 'Quinn and I have different roles' would have to be false, implying Liam shares Quinn’s role (spy), which is impossible since Quinn is the spy. So Liam must be a knight, and his statement is true (Quinn is Spy, Liam is Knight). Violet says: 'Among Liam and Yara, exactly one person is lying.' Since both Liam and Yara are knights and tell truth, exactly zero people are lying among them, making Violet’s statement false. Therefore Violet is a knave. Quinn’s statement: 'Violet, Yara, and I do not all have the same role.' With Violet=knave, Yara=knight, Quinn=spy, they do not all share the same role, so Quinn’s statement is true, which is fine for a spy. This assignment satisfies: exactly one spy (Quinn), knights tell truth (Yara, Liam), knave lies (Violet), and the statements are consistent with all given clues.

# Sample Return 4:
{
  "logical_order": ["Quinn is a spy", "Liam is a knight", "Violet is a knave", "Yara is a knight"]
}

<explanation>The logical order is clear in the reasoning.</explanation>
"""

# Response schema for the basic knight-knaves-spy game
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

def get_kks_system_prompt_with_confidence(num_player: int, include_confidence: bool = False) -> str:
    """Get the KKS system prompt, optionally including confidence scoring."""
    if not include_confidence:
        return kks_system_prompt.format(num_player=num_player)
    
    # The confidence instructions are already included in the base prompt
    return kks_system_prompt.format(num_player=num_player)

def get_kks_response_schema_with_confidence(include_confidence: bool = False) -> dict:
    """Get the KKS response schema, optionally including confidence field."""
    if not include_confidence:
        return kks_response_schema
    
    # Create a copy of the schema and add confidence field
    schema_with_confidence = kks_response_schema.copy()
    schema_with_confidence['properties']['confidence'] = {
        'type': 'integer',
        'minimum': 1,
        'maximum': 10,
        'description': 'Confidence level in the solution (1 to 10)'
    }
    schema_with_confidence['required'].append('confidence')
    
    return schema_with_confidence