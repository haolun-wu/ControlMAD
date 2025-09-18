#!/usr/bin/env python3
"""
Quick script to test a few models and save results to haolun_record.md
"""

import sys
import os
import json
import subprocess
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.config import test_config

# Thread lock for safe file operations
file_lock = threading.Lock()

# Quick test with selected models
QUICK_MODELS = [
    # OpenAI models
    {"provider": "openai", "model": "gpt-5-nano", "reasoning_effort": "medium", "name": "gpt-5-nano (medium effort)"},
    {"provider": "openai", "model": "gpt-5-nano", "reasoning_effort": "low", "name": "gpt-5-nano (low effort)"},
    {"provider": "openai", "model": "gpt-4o-mini", "name": "gpt-4o-mini"},
    {"provider": "openai", "model": "gpt-4.1-mini", "name": "gpt-4.1-mini"},
    
    # Gemini models
    {"provider": "gemini", "model": "gemini-2.5-flash-lite", "name": "gemini-2.5-flash-lite"},
    {"provider": "gemini", "model": "gemini-2.5-flash", "name": "gemini-2.5-flash"},
    
    # Ali models
    # {"provider": "ali", "model": "qwq-32b", "name": "qwq-32b"},
    {"provider": "ali", "model": "qwen-turbo-latest", "maximal_token": 4096, "thinking_budget": 4096, "name": "qwen-turbo-latest"},
    {"provider": "ali", "model": "qwen3-30b-a3b-thinking-2507", "maximal_token": 4096, "thinking_budget": 4096, "name": "qwen3-30b-a3b-thinking-2507"},
    {"provider": "ali", "model": "qwen-flash", "maximal_token": 4096, "thinking_budget": 4096, "name": "qwen-flash"},
    {"provider": "ali", "model": "qwen-plus", "maximal_token": 4096, "thinking_budget": 4096, "name": "qwen-plus"},
    
    # DeepSeek models (CST Cloud)
    # {"provider": "cstcloud", "model": "deepseek-reasoner", "name": "deepseek-reasoner"},
    
    # Claude models
    {"provider": "claude", "model": "claude-3-5-haiku-latest", "name": "claude-3-5-haiku-latest"},
]

def update_config_file(model_config):
    """Update utils/config.py with new model settings"""
    # Create model-specific output path directly in test folder
    model_name = model_config['name'].replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
    model_output_path = f"test/{model_name}_{test_config.game_size}.jsonl"
    
    config_content = f'''from .project_types import base_config, test_setup


game_config = base_config(
    name_pool=["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Hank", "Ivy", "Jack", "Kate", "Liam", "Mia", "Noah", "Olivia", "Peter", "Quinn", "Rachel", "Sam", "Tina", "Uma", "Violet", "Wendy", "Xavier", "Yara", "Zane"],
    role_map={{
        1: "knight",
        2: "knave",
        3: "spy"
    }},
    tf_map={{
        1: "telling the truth",
        2: "lying"
    }},
    eo_map={{
        1: "odd",
        2: "even"
    }},
    number_map={{
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten"
    }},
    game_size = {test_config.game_size},
    num_spy = 1,
    num_hint = 0
)

test_config = test_setup(
    game_size = {test_config.game_size},
    provider = "{model_config['provider']}",
    model = "{model_config['model']}",
    pass_at = {test_config.pass_at},
    num_worker = {test_config.num_worker},
    show_thinking = {test_config.show_thinking},
    groundtruth_path = "{test_config.groundtruth_path}",
    output_path = "{model_output_path}",
    enable_thinking = {test_config.enable_thinking},
    thinking_budget = {model_config.get('thinking_budget', test_config.thinking_budget)},
    stream = {test_config.stream},
    verbosity = "{test_config.verbosity}",
    reasoning_effort = "{model_config.get('reasoning_effort', test_config.reasoning_effort)}",
    reasoning_summary = "{test_config.reasoning_summary}",
    return_full_response = {test_config.return_full_response},
    truncation = {test_config.truncation},
    maximal_token = {model_config.get('maximal_token', test_config.maximal_token)}
)
'''
    
    with open(os.path.join(project_root, "utils/config.py"), "w") as f:
        f.write(config_content)

def run_baseline_test(model_config=None):
    """Run the baseline test and capture output"""
    try:
        if model_config:
            # Use command line arguments for model-specific testing
            model_name = model_config['model']
            provider = model_config['provider']
            model_name_clean = model_config['name'].replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
            output_path = f"test/{model_name_clean}_{test_config.game_size}.jsonl"
            
            cmd = [
                "conda", "run", "-n", "py10", "python", "test_baseline.py",
                model_name, provider, output_path
            ]
        else:
            cmd = ["conda", "run", "-n", "py10", "python", "test_baseline.py"]
            
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.path.join(project_root, "baseline")
        )
        return result.returncode == 0, result.stdout
    except Exception as e:
        return False, str(e)

def extract_metrics_from_output(output):
    """Extract metrics from the test output"""
    lines = output.split('\n')
    
    binary_accuracy = None
    smoother_accuracy = None
    avg_confidence = None
    
    for line in lines:
        if "Binary Accuracy:" in line and "correct answers" in line:
            # Extract percentage from line like "Binary Accuracy: 5/10 correct answers (50.0%)"
            try:
                percent_start = line.find('(') + 1
                percent_end = line.find('%')
                binary_accuracy = float(line[percent_start:percent_end])
            except:
                pass
        
        elif "Smoother Accuracy:" in line:
            # Extract percentage from line like "Smoother Accuracy: 74.0%"
            try:
                percent_start = line.find(':') + 1
                percent_end = line.find('%')
                smoother_accuracy = float(line[percent_start:percent_end].strip())
            except:
                pass
        
        elif "Average confidence:" in line:
            # Extract confidence from line like "Average confidence: 3.1/10"
            try:
                conf_start = line.find(':') + 1
                conf_end = line.find('/10')
                avg_confidence = float(line[conf_start:conf_end].strip())
            except:
                pass
    
    return binary_accuracy, smoother_accuracy, avg_confidence

def analyze_model_output_file(model_config):
    """Analyze the model-specific output file to compute statistics"""
    import json
    from utils.project_types import ground_truth
    
    # Create model-specific output path directly in test folder
    model_name = model_config['name'].replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
    output_file = os.path.join(project_root, f"test/{model_name}_{test_config.game_size}.jsonl")
    
    if not os.path.exists(output_file):
        print(f"⚠️  Output file not found: {output_file}")
        return None, None, None
    
    # Load ground truth
    groundtruth_path = os.path.join(project_root, f"{test_config.groundtruth_path}/{test_config.game_size}.jsonl")
    ground_truth_list = []
    
    try:
        with open(groundtruth_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    gt = ground_truth(
                        game_id=data['game_id'],
                        num_player=data['num_player'],
                        num_spy=data['num_spy'],
                        num_hint=data['num_hint'],
                        raw_schema=data['raw_schema'],
                        raw_statement=data['raw_statement'],
                        raw_solution=data['raw_solution'],
                        text_game=data['text_game'],
                        text_solution=data['text_solution']
                    )
                    ground_truth_list.append(gt)
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
    
    print(f"📊 Analyzing {len(responses)} responses for {model_config['name']}")
    
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
            if gt.game_id == game_id:
                current_gt = gt
                break
        
        if current_gt is None:
            continue
        
        total_matched += 1
        
        # Parse ground truth solution
        gt_pairs = _parse_groundtruth_solution(current_gt)
        
        # Parse LLM response
        llm_pairs, confidence = _parse_llm_response(llm_response)
        
        # Track confidence scores
        if confidence is not None:
            confidence_scores.append(confidence)
        
        # Compare answers
        if gt_pairs is not None and llm_pairs is not None:
            gt_set = set(gt_pairs)
            llm_set = set(llm_pairs)
            
            # Calculate player accuracy
            player_accuracy = _calculate_player_accuracy(gt_pairs, llm_pairs)
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

def _parse_groundtruth_solution(gt):
    """Parse ground truth solution to extract player-role pairs."""
    try:
        import re
        solution_text = gt.text_solution.strip()
        pairs = []
        
        # Pattern to match "PlayerName is a role."
        pattern = r'(\w+)\s+is\s+a\s+(knight|knave|spy)\.'
        matches = re.findall(pattern, solution_text, re.IGNORECASE)
        
        for player_name, role in matches:
            pairs.append((player_name, role.lower()))
        
        return pairs if pairs else None
    except Exception as e:
        print(f"Error parsing ground truth solution: {e}")
        return None

def _parse_llm_response(response_data):
    """Parse LLM response to extract player-role pairs and confidence score."""
    try:
        response_text = response_data.get('text', '')
        
        if not response_text or response_text == "ERROR":
            return None, None
        
        # Try to parse JSON from the response text
        try:
            llm_data = json.loads(response_text)
        except json.JSONDecodeError:
            # If direct JSON parsing fails, try to extract JSON pattern
            import re
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

def _calculate_player_accuracy(gt_pairs, llm_pairs):
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

def analyze_all_existing_models():
    """Analyze all models that have been tested and update the record with proper statistics"""
    print("🔍 Analyzing all existing model results...")
    
    # Look for model-specific files in test folder
    test_dir = os.path.join(project_root, "test")
    all_results = []
    
    if os.path.exists(test_dir):
        # Check for model-specific files in test folder
        for item in os.listdir(test_dir):
            if item.endswith(f'_{test_config.game_size}.jsonl') and os.path.isfile(os.path.join(test_dir, item)):
                # Extract model name from filename
                model_name = item.replace(f'_{test_config.game_size}.jsonl', '')
                output_file = os.path.join(test_dir, item)
                
                if os.path.exists(output_file):
                    print(f"\n📊 Analyzing {model_name}...")
                    
                    # Find the corresponding model config
                    model_config = None
                    for config in QUICK_MODELS:
                        config_model_name = config['name'].replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
                        if config_model_name == model_name:
                            model_config = config
                            break
                    
                    if model_config:
                        # Analyze this model's output
                        binary_acc, smoother_acc, avg_conf = analyze_model_output_file(model_config)
                        
                        if binary_acc is not None:
                            result = {
                                "model": model_config['name'],
                                "binary_accuracy": binary_acc,
                                "smoother_accuracy": smoother_acc,
                                "avg_confidence": avg_conf,
                                "success": True
                            }
                            all_results.append(result)
                            conf_str = f"{avg_conf:.1f}/10" if avg_conf is not None else "N/A"
                            print(f"   ✅ {model_config['name']}: Binary: {binary_acc:.1f}% | Smoother: {smoother_acc:.1f}% | Confidence: {conf_str}")
                        else:
                            print(f"   ❌ {model_name}: Could not analyze results")
                    else:
                        print(f"   ⚠️  {model_name}: No matching model config found")
        
        # Also check the main test/5.jsonl file if no model-specific files found
        if not all_results:
            main_output_file = os.path.join(test_dir, f"{test_config.game_size}.jsonl")
            if os.path.exists(main_output_file):
                print(f"\n📊 Analyzing main output file...")
                # Create a temporary model config for the main file
                temp_config = {"name": "Main Test Results"}
                binary_acc, smoother_acc, avg_conf = analyze_model_output_file(temp_config)
                
                if binary_acc is not None:
                    result = {
                        "model": "Main Test Results",
                        "binary_accuracy": binary_acc,
                        "smoother_accuracy": smoother_acc,
                        "avg_confidence": avg_conf,
                        "success": True
                    }
                    all_results.append(result)
                    print(f"   ✅ Main: Binary: {binary_acc:.1f}% | Smoother: {smoother_acc:.1f}% | Confidence: {avg_conf:.1f}/10")
    
    if all_results:
        print(f"\n📋 Found results for {len(all_results)} models")
        # Update the record with all results
        save_updated_record(all_results, test_config.game_size, test_config.truncation)
        print("✅ Record updated with proper statistics!")
    else:
        print("❌ No model results found to analyze")
    
    return all_results

def load_existing_record():
    """Load existing record file and parse it"""
    record_file = os.path.join(project_root, "baseline", "haolun_record.md")
    
    if not os.path.exists(record_file):
        return {}, None, None
    
    with open(record_file, 'r') as f:
        content = f.read()
    
    # Parse existing models
    existing_models = {}
    lines = content.split('\n')
    
    current_model = None
    current_data = {}
    
    for line in lines:
        line = line.strip()
        if line.startswith('## '):
            # Save previous model if exists
            if current_model and current_data:
                existing_models[current_model] = current_data.copy()
            
            # Start new model
            current_model = line[3:].strip()
            current_data = {}
        elif line.startswith('- Binary Accuracy:'):
            try:
                current_data['binary_accuracy'] = float(line.split(':')[1].split('%')[0].strip())
            except:
                pass
        elif line.startswith('- Smoother Accuracy:'):
            try:
                current_data['smoother_accuracy'] = float(line.split(':')[1].split('%')[0].strip())
            except:
                pass
        elif line.startswith('- Average Confidence:'):
            try:
                current_data['avg_confidence'] = float(line.split(':')[1].split('/')[0].strip())
            except:
                pass
        elif line.startswith('- Test Date:'):
            current_data['test_date'] = line.split(':', 1)[1].strip()
    
    # Save last model
    if current_model and current_data:
        existing_models[current_model] = current_data
    
    # Extract header info
    header_lines = []
    test_info = None
    for line in lines:
        if line.startswith('#') or line.startswith('Generated on:') or line.startswith('Test case:'):
            header_lines.append(line)
            if line.startswith('Test case:'):
                test_info = line
    
    return existing_models, header_lines, test_info

def save_updated_record(new_results, game_size, total_cases):
    """Save updated record file, replacing existing models and adding new ones"""
    record_file = os.path.join(project_root, "baseline", "haolun_record.md")
    
    # Load existing record
    existing_models, header_lines, test_info = load_existing_record()
    
    # Update with new results
    for result in new_results:
        if result.get("success", False):
            model_name = result["model"]
            existing_models[model_name] = {
                'binary_accuracy': result["binary_accuracy"],
                'smoother_accuracy': result["smoother_accuracy"],
                'avg_confidence': result["avg_confidence"],
                'test_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
    
    # Write updated record
    with open(record_file, "w") as f:
        # Write header
        if header_lines:
            for line in header_lines:
                f.write(line + "\n")
        else:
            f.write(f"# Test Results Record - Haolun\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Write test info
        if test_info:
            f.write(test_info + "\n\n")
        else:
            f.write(f"Test case: size {game_size}, game_id 1-{total_cases}\n\n")
        
        # Write all models (existing + new/updated)
        for model_name, data in existing_models.items():
            f.write(f"## {model_name}\n")
            if 'binary_accuracy' in data and data['binary_accuracy'] is not None:
                f.write(f"- Binary Accuracy: {data['binary_accuracy']:.1f}%\n")
            if 'smoother_accuracy' in data and data['smoother_accuracy'] is not None:
                f.write(f"- Smoother Accuracy: {data['smoother_accuracy']:.1f}%\n")
            if 'avg_confidence' in data and data['avg_confidence'] is not None:
                f.write(f"- Average Confidence: {data['avg_confidence']:.1f}/10\n")
            if 'test_date' in data:
                f.write(f"- Test Date: {data['test_date']}\n")
            f.write("\n")
    
    print(f"📁 Updated results saved to: haolun_record.md")

def test_single_model(model_config, model_index, total_models):
    """Test a single model and return results"""
    model_name = model_config["name"]
    print(f"\n🔄 Testing {model_index}/{total_models}: {model_name}")
    
    # Clear previous results - use model-specific output file directly in test folder
    model_name = model_config['name'].replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
    output_file = os.path.join(project_root, f"test/{model_name}_{test_config.game_size}.jsonl")
    
    # Create the test directory if it doesn't exist
    test_dir = os.path.join(project_root, "test")
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    # Remove existing file if it exists (overwrite)
    if os.path.exists(output_file):
        os.remove(output_file)
    
    # Run test
    success, output = run_baseline_test(model_config)
    
    if success:
        print(f"   ✅ {model_name}: Test completed")
        
        # Analyze the model-specific output file to get accurate statistics
        binary_acc, smoother_acc, avg_conf = analyze_model_output_file(model_config)
        
        if binary_acc is not None:
            smoother_str = f"{smoother_acc:.1f}%" if smoother_acc is not None else "N/A"
            conf_str = f"{avg_conf:.1f}/10" if avg_conf is not None else "N/A"
            print(f"   📊 {model_name}: Binary: {binary_acc:.1f}% | Smoother: {smoother_str} | Confidence: {conf_str}")
            
            return {
                "model": model_name,
                "binary_accuracy": binary_acc,
                "smoother_accuracy": smoother_acc,
                "avg_confidence": avg_conf,
                "success": True
            }
        else:
            print(f"   ❌ {model_name}: Could not analyze output file")
            return {"model": model_name, "success": False, "error": "Could not analyze output file"}
    else:
        print(f"   ❌ {model_name}: Test failed")
        print(f"   Error: {output}")
        return {"model": model_name, "success": False, "error": str(output)}

def main():
    """Main function with parallel execution"""
    print("🚀 Quick Multi-Model Testing (Parallel)")
    print(f"📊 Testing {len(QUICK_MODELS)} models in parallel")
    print("="*60)
    
    # Determine number of parallel workers (max 10 to avoid overwhelming APIs)
    max_workers = min(20, len(QUICK_MODELS))  # Increased from 10 to 20
    print(f"🔧 Using {max_workers} parallel workers")
    
    results = []
    
    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_model = {
            executor.submit(test_single_model, model_config, i, len(QUICK_MODELS)): model_config
            for i, model_config in enumerate(QUICK_MODELS, 1)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_model):
            model_config = future_to_model[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f"❌ {model_config['name']} generated an exception: {exc}")
                results.append({"model": model_config['name'], "success": False, "error": str(exc)})
    
    # Print summary
    print("\n" + "="*60)
    print("📋 SUMMARY OF ALL TESTS")
    print("="*60)
    
    successful_results = [r for r in results if r.get("success", False)]
    failed_results = [r for r in results if not r.get("success", False)]
    
    for result in successful_results:
        model_name = result["model"]
        binary_acc = result["binary_accuracy"]
        smoother_acc = result["smoother_accuracy"]
        avg_conf = result["avg_confidence"]
        
        smoother_str = f"{smoother_acc:.1f}%" if smoother_acc is not None else "N/A"
        conf_str = f"{avg_conf:.1f}/10" if avg_conf is not None else "N/A"
        print(f"✅ {model_name}: Binary: {binary_acc:.1f}% | Smoother: {smoother_str} | Confidence: {conf_str}")
    
    for result in failed_results:
        print(f"❌ {result['model']}: Failed - {result.get('error', 'Unknown error')}")
    
    # Save updated record (replaces existing models, adds new ones, keeps old ones)
    save_updated_record(results, test_config.game_size, test_config.truncation)
    
    print(f"🎉 Parallel testing completed! ({len(successful_results)}/{len(results)} successful)")

if __name__ == "__main__":
    import sys
    
    # Check if user wants to analyze existing results
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        analyze_all_existing_models()
    else:
        main()
