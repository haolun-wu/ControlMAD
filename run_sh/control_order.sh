#!/bin/bash

# Get the directory where this script is located and change to parent directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Define LLM configurations for the debate
LLM_CONFIGS='[
    {
        "provider": "openai", 
        "model": "gpt-5-nano",
        "reasoning_effort": "medium"
    },
    {
        "provider": "ali",
        "model": "qwen-turbo-latest"
    },
    {
        "provider": "gemini",
        "model": "gemini-2.5-flash-lite"
    }
]'

# Run the debate with flexible configuration using key=value format
# debate_order_control=0: Use ground truth order (default)
# debate_order_control=1: Use GPT-5 agent to decide the debate order after initialization
python run_debate_chat.py flexible llm_configs="$LLM_CONFIGS" game_size=4 game_id_range=1,100 self_reported_confidence=false script_name="control_order" debate_order_control=1
python run_debate_chat.py flexible llm_configs="$LLM_CONFIGS" game_size=6 game_id_range=1,100 self_reported_confidence=false script_name="control_order" debate_order_control=1
python run_debate_chat.py flexible llm_configs="$LLM_CONFIGS" game_size=8 game_id_range=1,100 self_reported_confidence=false script_name="control_order" debate_order_control=1