#!/bin/bash

# Define LLM configurations for the debate
LLM_CONFIGS='[
    {
        "provider": "openai", 
        "model": "gpt-5-nano",
        "reasoning_effort": "low"
    },
    {
        "provider": "gemini",
        "model": "gemini-2.5-flash-lite"
    },
    {
        "provider": "ali",
        "model": "qwen-turbo-latest"
    }
]'

# Run the debate with flexible configuration using key=value format
python run_debate.py flexible llm_configs="$LLM_CONFIGS" game_size=4 game_id_range=1,1 self_reported_confidence=false

# Visualization
# python debate/debate_visualizer.py --results-dir ./debate_results/agent3_gpt-5-nano_gemini-2.5-flash-lite_qwen-turbo-latest