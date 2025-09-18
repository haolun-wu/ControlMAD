#!/bin/bash

# Define LLM configurations for the debate
LLM_CONFIGS='[
    {
        "provider": "openai", 
        "model": "o3-mini"
    },
    {
        "provider": "ali",
        "model": "qwen-flash"
    },
    {
        "provider": "gemini",
        "model": "gemini-2.5-flash-lite"
    }
]'

# Run the debate with flexible configuration using key=value format
python run_debate.py flexible llm_configs="$LLM_CONFIGS" game_size=4 game_id_range=1,100 self_reported_confidence=false
python run_debate.py flexible llm_configs="$LLM_CONFIGS" game_size=4 game_id_range=1,100 self_reported_confidence=true
python run_debate.py flexible llm_configs="$LLM_CONFIGS" game_size=6 game_id_range=1,100 self_reported_confidence=false
python run_debate.py flexible llm_configs="$LLM_CONFIGS" game_size=6 game_id_range=1,100 self_reported_confidence=true
python run_debate.py flexible llm_configs="$LLM_CONFIGS" game_size=9 game_id_range=1,100 self_reported_confidence=false
python run_debate.py flexible llm_configs="$LLM_CONFIGS" game_size=9 game_id_range=1,100 self_reported_confidence=true

