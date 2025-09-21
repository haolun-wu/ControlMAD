#!/bin/bash

# Script to generate visualizations for all debate results using debate_visualizer.py
# Usage: ./visualize_all_results.sh [base_path]

BASE_PATH=${1:-"/Users/haolunwu/Documents/GitHub/ControlMAD/debate_results/agent3_gpt-5-nano_qwen-turbo-latest_gemini-2.5-flash-lite_conf_true"}

echo "🎨 Generating visualizations for chat debate results..."
echo "📁 Base path: $BASE_PATH"

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

# Process each directory containing JSON files
echo "$JSON_FILES" | while read -r json_file; do
    if [ -n "$json_file" ]; then
        # Get the directory containing the JSON file
        json_dir=$(dirname "$json_file")
        json_filename=$(basename "$json_file")
        
        echo "📊 Processing: $json_filename in $(basename "$json_dir")"
        
        # Run the debate visualizer for this directory
        cd /Users/haolunwu/Documents/GitHub/ControlMAD && python debate/chat/debate_visualizer.py --results-dir "$json_dir" --output-dir "$json_dir"
        
        if [ $? -eq 0 ]; then
            echo "✅ Generated visualizations for $json_filename"
        else
            echo "❌ Failed to generate visualizations for $json_filename"
        fi
        echo ""
    fi
done

echo "✅ Visualization generation completed!"
