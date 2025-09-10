# KKS-bench

## Game setup
- Three roles:
    - Knights: always telling the truth.
    - Knaves: always lying.
    - Spy: figure crossed.
- Game manager: provide hints. One fixed hint is the number of spies in the game.
- Goal:
    - Each player provides sentence, and the truthfulness align with their roles. The goal is to deduce the roles of all players.

## Codebase structure
- project_types.py
    - save the custom types of the codebase.
- config.py
    - store the configuration of the codebase:
        - game_config: configuration for the groundtruth generation.
        - test_config: configuration for the tests.
- utility.py
    - store help functions for the whole codebase.
- generator.py
    - generate abstract game instance and render it with texts.
- validator.py
    - validate the abstract game instance to make sure that there is a unque solution.
- groundTruth.py
    - generate the ground truth.
- prompts.py
    - store system prompts and output schema.
- test_baseline.py
    - test for the zero-shot of LLMs.

## Folders
- groundtruth: storing all groundtruth of the game.
    - groundtruth are grouped by game size. the record.json helps to generate unique game id.
- test: storing all test results.

