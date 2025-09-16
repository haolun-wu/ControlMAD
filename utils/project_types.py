from dataclasses import dataclass
from typing import List, Dict

@dataclass
class player:
    name: str
    statement: str
    description: str = ""

@dataclass
class base_config:
    name_pool: List[str]
    role_map: Dict[int, str]
    tf_map: Dict[int, str]
    eo_map: Dict[int, str]
    number_map: Dict[int, str]
    game_size: int
    num_spy: int
    num_hint: int

@dataclass
class ground_truth:
    game_id: int
    num_player: int
    num_spy: int
    num_hint: int
    raw_schema: list[int]
    raw_statement: list
    raw_solution: list[int]
    text_game: str
    text_solution: str

@dataclass
class test_setup:
    game_size: int
    provider: str
    model: str
    pass_at: int
    num_worker: int
    show_thinking: bool
    groundtruth_path: str
    output_path: str
    enable_thinking: bool
    stream: bool
    verbosity: str
    reasoning_effort: str
    reasoning_summary: str
    return_full_response: bool
    truncation: int

@dataclass
class token_usage:
    input: int
    output: int
    reasoning: int
    cached: int
    total: int

@dataclass
class response_format:
    text: str
    usage: token_usage
    summary: str
    error: str



