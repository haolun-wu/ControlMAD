#!/bin/bash

# Get the directory where this script is located and change to parent directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Define LLM configurations for the debate
LLM_CONFIGS='[
    {
        "provider": "gemini", 
        "model": "gemini-2.5-flash"
    },
    {
        "provider": "ali",
        "model": "qwen-flash"
    },
    {
        "provider": "openai",
        "model": "gpt-4.1-mini"
    }
]'

# Run the debate with flexible configuration using key=value format
python run_debate_chat.py flexible llm_configs="$LLM_CONFIGS" game_size=4 game_id_range=1,50 self_reported_confidence=false script_name="mix_C"
python run_debate_chat.py flexible llm_configs="$LLM_CONFIGS" game_size=6 game_id_range=1,50 self_reported_confidence=false script_name="mix_C"
python run_debate_chat.py flexible llm_configs="$LLM_CONFIGS" game_size=8 game_id_range=1,50 self_reported_confidence=false script_name="mix_C"
