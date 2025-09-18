#!/usr/bin/env python3
"""
Standalone script to analyze existing test results and update haolun_record.md
"""

import os
import json
import re
from datetime import datetime

# Project root
project_root = "/Users/haolunwu/Documents/GitHub/ControlMAD"

def parse_groundtruth_solution(text_solution):
    """Parse ground truth solution to extract player-role pairs."""
    try:
        pairs = []
        # Pattern to match "PlayerName is a role."
        pattern = r'(\w+)\s+is\s+a\s+(knight|knave|spy)\.'
        matches = re.findall(pattern, text_solution, re.IGNORECASE)
        
        for player_name, role in matches:
            pairs.append((player_name, role.lower()))
        
        return pairs if pairs else None
    except Exception as e:
        print(f"Error parsing ground truth solution: {e}")
        return None

def parse_llm_response(response_text):
    """Parse LLM response to extract player-role pairs and confidence score."""
    try:
        if not response_text or response_text == "ERROR":
            return None, None
        
        # Try to parse JSON from the response text
        try:
            llm_data = json.loads(response_text)
        except json.JSONDecodeError:
            # If direct JSON parsing fails, try to extract JSON pattern
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                llm_data = json.loads(json_str)
            else:
                return None, None
        
        # Parse the JSON based on expected format
        pairs = []
        confidence = None
        
        # Extract confidence score if present
        if "confidence" in llm_data:
            confidence = llm_data["confidence"]
            if isinstance(confidence, (int, float)):
                confidence = float(confidence)
                if confidence < 1 or confidence > 10:
                    confidence = None
            else:
                confidence = None
        
        # Check if it's the new format with "players" array
        if "players" in llm_data and isinstance(llm_data["players"], list):
            for player_data in llm_data["players"]:
                if isinstance(player_data, dict) and "name" in player_data and "role" in player_data:
                    pairs.append((player_data["name"], player_data["role"]))
        else:
            # Old format - direct name: role mapping
            for key, value in llm_data.items():
                if key not in ["reasoning", "confidence"] and isinstance(value, str):
                    pairs.append((key, value))
        
        return pairs, confidence
        
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        return None, None

def calculate_player_accuracy(gt_pairs, llm_pairs):
    """Calculate the percentage of players that were predicted correctly."""
    if not gt_pairs or not llm_pairs:
        return 0.0
    
    # Convert to dictionaries for easier lookup
    gt_dict = dict(gt_pairs)
    llm_dict = dict(llm_pairs)
    
    # Get all unique players from both ground truth and LLM response
    all_players = set(gt_dict.keys()) | set(llm_dict.keys())
    
    if not all_players:
        return 0.0
    
    correct_predictions = 0
    total_players = len(all_players)
    
    for player in all_players:
        gt_role = gt_dict.get(player)
        llm_role = llm_dict.get(player)
        
        # If both roles exist and match, it's correct
        if gt_role and llm_role and gt_role == llm_role:
            correct_predictions += 1
    
    return correct_predictions / total_players if total_players > 0 else 0.0

def analyze_output_file(output_file, groundtruth_file):
    """Analyze a specific output file and compute statistics"""
    
    # Load ground truth
    ground_truth_list = []
    try:
        with open(groundtruth_file, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    ground_truth_list.append(data)
                except Exception as e:
                    print(f"Warning: Error parsing ground truth line {line_num}: {e}")
                    continue
    except Exception as e:
        print(f"Error loading ground truth: {e}")
        return None, None, None
    
    # Load responses
    responses = []
    try:
        with open(output_file, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    response_data = json.loads(line)
                    responses.append(response_data)
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON in response file on line {line_num}: {e}")
                    continue
    except Exception as e:
        print(f"Error reading response file: {e}")
        return None, None, None
    
    print(f"📊 Analyzing {len(responses)} responses")
    
    # Compute statistics
    correct_answers = 0
    total_matched = 0
    confidence_scores = []
    total_player_accuracy = 0.0
    
    for response_data in responses:
        if "game_id" not in response_data or "llm" not in response_data:
            continue
        
        game_id = response_data["game_id"]
        llm_response = response_data["llm"]
        
        # Find corresponding ground truth
        current_gt = None
        for gt in ground_truth_list:
            if gt['game_id'] == game_id:
                current_gt = gt
                break
        
        if current_gt is None:
            continue
        
        total_matched += 1
        
        # Parse ground truth solution
        gt_pairs = parse_groundtruth_solution(current_gt['text_solution'])
        
        # Parse LLM response
        llm_pairs, confidence = parse_llm_response(llm_response.get('text', ''))
        
        # Track confidence scores
        if confidence is not None:
            confidence_scores.append(confidence)
        
        # Compare answers
        if gt_pairs is not None and llm_pairs is not None:
            gt_set = set(gt_pairs)
            llm_set = set(llm_pairs)
            
            # Calculate player accuracy
            player_accuracy = calculate_player_accuracy(gt_pairs, llm_pairs)
            total_player_accuracy += player_accuracy
            
            if gt_set == llm_set:
                correct_answers += 1
    
    # Calculate final metrics
    binary_accuracy = 100 * correct_answers / total_matched if total_matched > 0 else 0
    smoother_accuracy = 100 * total_player_accuracy / total_matched if total_matched > 0 else 0
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else None
    
    print(f"   ✅ Binary Accuracy: {correct_answers}/{total_matched} ({binary_accuracy:.1f}%)")
    print(f"   📊 Smoother Accuracy: {smoother_accuracy:.1f}%")
    print(f"   🎯 Average Confidence: {avg_confidence:.1f}/10" if avg_confidence else "   ⚠️ No confidence scores")
    
    return binary_accuracy, smoother_accuracy, avg_confidence

def update_record_file(results):
    """Update the haolun_record.md file with proper statistics"""
    record_file = os.path.join(project_root, "baseline", "haolun_record.md")
    
    # Create new content
    content = f"""# Test Results Record - Haolun
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Test case: size 5, game_id 1-100

"""
    
    for result in results:
        content += f"## {result['model']}\n"
        content += f"- Binary Accuracy: {result['binary_accuracy']:.1f}%\n"
        content += f"- Smoother Accuracy: {result['smoother_accuracy']:.1f}%\n"
        if result['avg_confidence'] is not None:
            content += f"- Average Confidence: {result['avg_confidence']:.1f}/10\n"
        content += f"- Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Write the file
    with open(record_file, 'w') as f:
        f.write(content)
    
    print(f"📁 Updated results saved to: haolun_record.md")

def main():
    """Main function to analyze existing results"""
    print("🔍 Analyzing existing test results...")
    
    # File paths
    groundtruth_file = os.path.join(project_root, "groundtruth/5.jsonl")
    
    if not os.path.exists(groundtruth_file):
        print(f"❌ Ground truth file not found: {groundtruth_file}")
        return
    
    # Look for model-specific output files in test folder
    test_dir = os.path.join(project_root, "test")
    all_results = []
    
    if os.path.exists(test_dir):
        # Check for model-specific files in test folder
        for item in os.listdir(test_dir):
            if item.endswith('_5.jsonl') and os.path.isfile(os.path.join(test_dir, item)):
                # Extract model name from filename
                model_name = item.replace('_5.jsonl', '')
                output_file = os.path.join(test_dir, item)
                
                if os.path.exists(output_file):
                    print(f"\n📊 Analyzing {model_name}...")
                    binary_acc, smoother_acc, avg_conf = analyze_output_file(output_file, groundtruth_file)
                    
                    if binary_acc is not None:
                        result = {
                            "model": model_name.replace('_', ' ').replace('  ', ' '),
                            "binary_accuracy": binary_acc,
                            "smoother_accuracy": smoother_acc,
                            "avg_confidence": avg_conf
                        }
                        all_results.append(result)
                        conf_str = f"{avg_conf:.1f}/10" if avg_conf is not None else "N/A"
                        print(f"   ✅ {model_name}: Binary: {binary_acc:.1f}% | Smoother: {smoother_acc:.1f}% | Confidence: {conf_str}")
                    else:
                        print(f"   ❌ {model_name}: Could not analyze results")
        
        # Also check the main test/5.jsonl file if no model-specific files found
        if not all_results:
            main_output_file = os.path.join(test_dir, "5.jsonl")
            if os.path.exists(main_output_file):
                print(f"\n📊 Analyzing main output file...")
                binary_acc, smoother_acc, avg_conf = analyze_output_file(main_output_file, groundtruth_file)
                
                if binary_acc is not None:
                    result = {
                        "model": "Main Test Results",
                        "binary_accuracy": binary_acc,
                        "smoother_accuracy": smoother_acc,
                        "avg_confidence": avg_conf
                    }
                    all_results.append(result)
                    conf_str = f"{avg_conf:.1f}/10" if avg_conf is not None else "N/A"
                    print(f"   ✅ Main: Binary: {binary_acc:.1f}% | Smoother: {smoother_acc:.1f}% | Confidence: {conf_str}")
    
    if all_results:
        # Update the record with all results
        update_record_file(all_results)
        print(f"\n✅ Analysis complete! Found results for {len(all_results)} models")
    else:
        print("❌ No model results found to analyze")

if __name__ == "__main__":
    main()
