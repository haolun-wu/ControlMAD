#!/bin/bash

# Script to generate visualizations for all debate results using debate_visualizer.py
# Usage: ./visualize_all_results.sh [base_path] [max_processes]
# 
# Examples:
#   ./visualize_all_results.sh                                    # Use default path and auto-detect processes
#   ./visualize_all_results.sh /path/to/results                   # Use custom path, auto-detect processes
#   ./visualize_all_results.sh /path/to/results 4                 # Use custom path and 4 processes
#   ./visualize_all_results.sh "" 1                               # Use default path but force sequential (1 process)

BASE_PATH=${1:-"/Users/haolunwu/Documents/GitHub/ControlMAD/debate_results/agent3_gpt-5-nano_qwen-turbo-latest_gemini-2.5-flash-lite_conf_true"}
MAX_PROCESSES=${2:-""}

echo "🎨 Generating visualizations for chat debate results..."
echo "📁 Base path: $BASE_PATH"
if [ -n "$MAX_PROCESSES" ]; then
    echo "⚙️  Max processes: $MAX_PROCESSES"
else
    echo "⚙️  Max processes: auto-detect"
fi

# Find all debate_session*.json files in subdirectories
echo "🔍 Searching for debate session files..."
JSON_FILES=$(find "$BASE_PATH" -name "debate_session*.json" -type f | sort)

if [ -z "$JSON_FILES" ]; then
    echo "❌ No debate_session*.json files found in $BASE_PATH"
    exit 1
fi

echo "📁 Found $(echo "$JSON_FILES" | wc -l) debate session files:"
echo "$JSON_FILES" | while read -r file; do
    echo "  - $(basename "$file")"
done

echo ""
echo "🎨 Generating visualizations..."

# Get unique directories containing JSON files
UNIQUE_DIRS=$(echo "$JSON_FILES" | xargs -I {} dirname {} | sort -u)
TOTAL_DIRS=$(echo "$UNIQUE_DIRS" | wc -l)

echo "📁 Found $TOTAL_DIRS unique directories to process"

# Determine parallel processing strategy
if [ -n "$MAX_PROCESSES" ]; then
    PARALLEL_JOBS=$MAX_PROCESSES
    echo "🚀 Using cross-directory parallel processing with $PARALLEL_JOBS processes"
else
    # Auto-detect: use min of CPU count and number of directories
    CPU_COUNT=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    PARALLEL_JOBS=$(($CPU_COUNT < $TOTAL_DIRS ? $CPU_COUNT : $TOTAL_DIRS))
    echo "🚀 Using cross-directory parallel processing with $PARALLEL_JOBS processes (auto-detected)"
fi

# Function to process a single directory
process_directory() {
    local json_dir="$1"
    local json_filename=$(basename "$(find "$json_dir" -name "debate_session*.json" | head -1)")
    
    echo "📊 Processing: $json_filename in $(basename "$json_dir")"
    
    # Build the command with optional max-processes parameter
    CMD="python debate/chat/debate_visualizer.py --results-dir \"$json_dir\" --output-dir \"$json_dir\""
    if [ -n "$MAX_PROCESSES" ]; then
        CMD="$CMD --max-processes 1"  # Force sequential within directory since only 1 JSON
    fi
    
    # Run the debate visualizer for this directory
    cd /Users/haolunwu/Documents/GitHub/ControlMAD && eval $CMD
    
    if [ $? -eq 0 ]; then
        echo "✅ Generated visualizations for $json_filename"
    else
        echo "❌ Failed to generate visualizations for $json_filename"
    fi
}

# Export the function so it can be used by parallel
export -f process_directory
export MAX_PROCESSES

# Process directories in parallel
echo "$UNIQUE_DIRS" | xargs -P "$PARALLEL_JOBS" -I {} bash -c 'process_directory "{}"'

echo "✅ Visualization generation completed!"
