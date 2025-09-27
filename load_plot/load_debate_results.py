#!/usr/bin/env python3
"""
Script to load debate results and calculate accuracies for different game sizes.
Returns strict accuracy (1 if fully correct, 0 otherwise) and smooth accuracy 
(partial credit based on individual player role correctness).
"""

import json
import os
import glob
from typing import Dict, List, Tuple
import re

def load_debate_results(results_dir: str) -> Dict[int, Dict[str, Dict]]:
    """
    Load all debate results from the specified directory.
    
    Args:
        results_dir: Path to the debate results directory
        
    Returns:
        Dictionary with structure: {game_size: {game_id: {data}}}
    """
    results = {}
    
    # Get all game directories
    game_dirs = glob.glob(os.path.join(results_dir, "game_size*"))
    
    for game_dir in game_dirs:
        # Extract game size and game id from directory name
        dir_name = os.path.basename(game_dir)
        match = re.match(r'game_size(\d+)_id(\d+)', dir_name)
        if not match:
            continue
            
        game_size = int(match.group(1))
        game_id = int(match.group(2))
        
        # Find the debate session file
        session_files = glob.glob(os.path.join(game_dir, "debate_session_*.json"))
        if not session_files:
            print(f"Warning: No debate session file found in {game_dir}")
            continue
            
        session_file = session_files[0]  # Take the first (should be only) one
        
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if game_size not in results:
                results[game_size] = {}
            results[game_size][game_id] = data
            
        except Exception as e:
            print(f"Error loading {session_file}: {e}")
            continue
    
    return results

def calculate_strict_accuracy(ground_truth: Dict[str, str], final_solution: Dict[str, str]) -> float:
    """
    Calculate strict accuracy: 1 if fully correct, 0 otherwise.
    
    Args:
        ground_truth: Ground truth role assignments
        final_solution: Final solution from the debate
        
    Returns:
        Strict accuracy (0.0 or 1.0)
    """
    if ground_truth == final_solution:
        return 1.0
    return 0.0

def calculate_smooth_accuracy(ground_truth: Dict[str, str], final_solution: Dict[str, str]) -> float:
    """
    Calculate smooth accuracy: partial credit based on individual player role correctness.
    
    Args:
        ground_truth: Ground truth role assignments
        final_solution: Final solution from the debate
        
    Returns:
        Smooth accuracy (0.0 to 1.0)
    """
    if not ground_truth or not final_solution:
        return 0.0
    
    correct_assignments = 0
    total_assignments = len(ground_truth)
    
    for player, correct_role in ground_truth.items():
        if player in final_solution and final_solution[player] == correct_role:
            correct_assignments += 1
    
    return correct_assignments / total_assignments

def extract_game_results(results: Dict[int, Dict[str, Dict]]) -> Dict[int, Tuple[List[float], List[float]]]:
    """
    Extract game results and calculate accuracies for each game size.
    
    Args:
        results: Loaded debate results
        
    Returns:
        Dictionary with structure: {game_size: (strict_accuracies, smooth_accuracies)}
    """
    game_results = {}
    
    for game_size in sorted(results.keys()):
        strict_accuracies = []
        smooth_accuracies = []
        
        # Get all game IDs for this game size and sort them
        game_ids = sorted(results[game_size].keys())
        
        for game_id in game_ids:
            game_data = results[game_size][game_id]
            
            # Extract ground truth and final solution
            ground_truth = game_data.get('ground_truth_solution', {})
            final_solution = game_data.get('final_vote', {})
            
            # Calculate accuracies
            strict_acc = calculate_strict_accuracy(ground_truth, final_solution)
            smooth_acc = calculate_smooth_accuracy(ground_truth, final_solution)
            
            strict_accuracies.append(strict_acc)
            smooth_accuracies.append(smooth_acc)
        
        game_results[game_size] = (strict_accuracies, smooth_accuracies)
    
    return game_results

def save_results_to_file(game_results: Dict[int, Tuple[List[float], List[float]]], output_file: str):
    """
    Save the results to a JSON file.
    
    Args:
        game_results: Dictionary with game results
        output_file: Path to output file
    """
    # Convert to serializable format
    serializable_results = {}
    for game_size, (strict_accs, smooth_accs) in game_results.items():
        serializable_results[str(game_size)] = {
            'strict_accuracies': strict_accs,
            'smooth_accuracies': smooth_accs,
            'num_games': len(strict_accs),
            'strict_mean': sum(strict_accs) / len(strict_accs) if strict_accs else 0,
            'smooth_mean': sum(smooth_accs) / len(smooth_accs) if smooth_accs else 0
        }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to {output_file}")

def main():
    """Main function to load and analyze debate results."""
    # Path to the debate results directory
    results_dir = "/Users/haolunwu/Documents/GitHub/ControlMAD/debate_results/control_depth_agent3_gpt-5-nano_qwen-turbo-latest_gemini-2.5-flash-lite_conf_false"
    
    print("Loading debate results...")
    results = load_debate_results(results_dir)
    
    print(f"Loaded results for game sizes: {sorted(results.keys())}")
    for game_size in sorted(results.keys()):
        print(f"  Game size {game_size}: {len(results[game_size])} games")
    
    print("\nCalculating accuracies...")
    game_results = extract_game_results(results)
    
    print("\nResults by game size:")
    for game_size in sorted(game_results.keys()):
        strict_accs, smooth_accs = game_results[game_size]
        
        print(f"\nGame Size {game_size}:")
        print(f"  Number of games: {len(strict_accs)}")
        print(f"  Strict accuracy - Mean: {sum(strict_accs)/len(strict_accs):.3f}, "
              f"Min: {min(strict_accs):.3f}, Max: {max(strict_accs):.3f}")
        print(f"  Smooth accuracy - Mean: {sum(smooth_accs)/len(smooth_accs):.3f}, "
              f"Min: {min(smooth_accs):.3f}, Max: {max(smooth_accs):.3f}")
        
        # Show first few accuracies as examples
        print(f"  First 5 strict accuracies: {strict_accs[:5]}")
        print(f"  First 5 smooth accuracies: {[f'{x:.3f}' for x in smooth_accs[:5]]}")
    
    # Save results to file
    output_file = "/Users/haolunwu/Documents/GitHub/ControlMAD/load_plot/debate_results_analysis.json"
    save_results_to_file(game_results, output_file)
    
    return game_results

if __name__ == "__main__":
    results = main()
    
    # Return the results in the requested format
    print("\n" + "="*50)
    print("FINAL RESULTS (as requested):")
    print("="*50)
    
    for game_size in sorted(results.keys()):
        strict_accs, smooth_accs = results[game_size]
        print(f"\nGame Size {game_size}:")
        print(f"Strict Accuracies ({len(strict_accs)} values): {strict_accs}")
        print(f"Smooth Accuracies ({len(smooth_accs)} values): {smooth_accs}")