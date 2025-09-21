import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import os
import json
import argparse
from datetime import datetime

import sys
# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from debate.chat.debate_system_chat import DebateSession, DebateRound, AgentResponse

class ChatHistoryDebateVisualizer:
    """Visualization system for chat history multi-agent debate performance."""
    
    def __init__(self, output_path: str = "./chat_debate_visualizations"):
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def create_performance_matrix(self, sessions: List[DebateSession], 
                                save_path: Optional[str] = None) -> str:
        """Create a performance matrix showing accuracy across agents and rounds."""
        
        # Collect data for all phases
        data = []
        
        for session in sessions:
            # Initial proposals
            for proposal in session.initial_proposals:
                correct = 0
                total = len(session.ground_truth_solution)
                for player, role in session.ground_truth_solution.items():
                    if proposal.player_role_assignments.get(player) == role:
                        correct += 1
                accuracy = correct / total if total > 0 else 0
                
                data.append({
                    'Game': session.game_id,
                    'Agent': proposal.agent_name,
                    'Phase': 'Initial',
                    'Accuracy': accuracy
                })
            
            # Debate phases for each round
            for round_data in session.debate_rounds:
                for response in round_data.agent_responses:
                    if response.phase in ['debate', 'self_adjustment']:
                        correct = 0
                        total = len(session.ground_truth_solution)
                        for player, role in session.ground_truth_solution.items():
                            if response.player_role_assignments.get(player) == role:
                                correct += 1
                        accuracy = correct / total if total > 0 else 0
                        
                        # Only include self-adjustment phases for the visualization
                        # This represents the state after each round's self-adjustment
                        if response.phase == 'self_adjustment':
                            data.append({
                                'Game': session.game_id,
                                'Agent': response.agent_name,
                                'Phase': f'After Round {round_data.round_number}',
                                'Accuracy': accuracy
                            })
            
            # Final consensus (if available)
            if session.final_vote:
                correct = 0
                total = len(session.ground_truth_solution)
                for player, role in session.ground_truth_solution.items():
                    if session.final_vote.get(player) == role:
                        correct += 1
                accuracy = correct / total if total > 0 else 0
                
                # Add final consensus for all agents (they should all have the same result)
                for proposal in session.initial_proposals:
                    data.append({
                        'Game': session.game_id,
                        'Agent': proposal.agent_name,
                        'Phase': 'Final Consensus',
                        'Accuracy': accuracy
                    })
        
        df = pd.DataFrame(data)
        
        # Create pivot table for heatmap
        pivot_table = df.pivot_table(
            values='Accuracy', 
            index='Agent', 
            columns='Phase', 
            aggfunc='mean'
        )
        
        # Reorder columns for logical flow
        # Extract round numbers and sort
        phase_columns = list(pivot_table.columns)
        initial_col = 'Initial'
        round_cols = [col for col in phase_columns if col.startswith('After Round')]
        final_consensus_col = 'Final Consensus' if 'Final Consensus' in phase_columns else None
        
        # Sort round columns by round number
        def extract_round_number(col):
            if col.startswith('After Round'):
                return int(col.split(' ')[2])
            return 0
        
        round_cols.sort(key=extract_round_number)
        
        # Build phase order: Initial -> Rounds -> Final Consensus
        phase_order = [initial_col] + round_cols
        if final_consensus_col:
            phase_order.append(final_consensus_col)
        
        pivot_table = pivot_table.reindex(columns=phase_order)
        
        # Create heatmap with light purple theme
        plt.figure(figsize=(max(8, len(phase_order) * 2), 6))
        
        # Create custom light purple colormap (light purple for low accuracy, dark purple for high accuracy)
        from matplotlib.colors import LinearSegmentedColormap
        colors = ['#E6E6FA', '#D8BFD8', '#DDA0DD', '#DA70D6', '#BA55D3', '#9370DB', '#8A2BE2', '#7B68EE', '#6A5ACD', '#483D8B']
        purple_cmap = LinearSegmentedColormap.from_list('light_purple', colors, N=256)
        
        sns.heatmap(pivot_table, annot=True, cmap=purple_cmap, 
                   vmin=0, vmax=1, fmt='.2f', cbar_kws={'label': 'Accuracy'})
        plt.title('Agent Performance Matrix: Accuracy by Round\n(Shows how each agent\'s accuracy changes through debate rounds and final consensus)')
        plt.xlabel('Phase')
        plt.ylabel('Agent')
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_path, "performance_matrix.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Performance matrix saved to: {save_path}")
        return save_path
    
    def create_consensus_tracking(self, sessions: List[DebateSession],
                                save_path: Optional[str] = None) -> str:
        """Create a plot showing consensus evolution over rounds."""
        
        # Collect consensus data
        consensus_data = []
        for session in sessions:
            for round_data in session.debate_rounds:
                # Count votes for the target player
                votes = {}
                for response in round_data.agent_responses:
                    player = round_data.player_name
                    if player in response.player_role_assignments:
                        role = response.player_role_assignments[player]
                        votes[role] = votes.get(role, 0) + 1
                
                # Calculate consensus metrics
                total_votes = sum(votes.values())
                max_votes = max(votes.values()) if votes else 0
                consensus_ratio = max_votes / total_votes if total_votes > 0 else 0
                
                consensus_data.append({
                    'Game': session.game_id,
                    'Round': round_data.round_number,
                    'Player': round_data.player_name,
                    'Consensus_Ratio': consensus_ratio,
                    'Total_Votes': total_votes,
                    'Max_Votes': max_votes
                })
        
        df = pd.DataFrame(consensus_data)
        
        # Create line plot
        plt.figure(figsize=(12, 6))
        
        # Plot consensus ratio over rounds
        for game_id in df['Game'].unique():
            game_data = df[df['Game'] == game_id]
            plt.plot(game_data['Round'], game_data['Consensus_Ratio'], 
                    marker='o', label=f'Game {game_id}', alpha=0.7)
        
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Majority Threshold')
        plt.xlabel('Round Number')
        plt.ylabel('Consensus Ratio')
        plt.title('Consensus Evolution Over Debate Rounds')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_path, "consensus_tracking.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Consensus tracking saved to: {save_path}")
        return save_path
    
    def create_agent_comparison(self, sessions: List[DebateSession],
                              save_path: Optional[str] = None) -> str:
        """Create a comparison chart of agent performance."""
        
        # Collect agent performance data
        agent_data = {}
        for session in sessions:
            for round_data in session.debate_rounds:
                for response in round_data.agent_responses:
                    agent_name = response.agent_name
                    if agent_name not in agent_data:
                        agent_data[agent_name] = []
                    
                    # Calculate accuracy
                    correct = 0
                    total = len(session.ground_truth_solution)
                    for player, role in session.ground_truth_solution.items():
                        if response.player_role_assignments.get(player) == role:
                            correct += 1
                    accuracy = correct / total if total > 0 else 0
                    
                    agent_data[agent_name].append({
                        'Accuracy': accuracy,
                        'Phase': response.phase,
                        'Round': round_data.round_number
                    })
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Agent Performance Comparison', fontsize=16)
        
        # 1. Box plot of accuracy by agent
        agent_names = list(agent_data.keys())
        accuracy_data = [agent_data[name] for name in agent_names]
        accuracy_values = [[item['Accuracy'] for item in data] for data in accuracy_data]
        
        axes[0, 0].boxplot(accuracy_values, labels=agent_names)
        axes[0, 0].set_title('Accuracy Distribution by Agent')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Accuracy by phase
        phase_data = {}
        for agent_name, data in agent_data.items():
            for item in data:
                phase = item['Phase']
                if phase not in phase_data:
                    phase_data[phase] = []
                phase_data[phase].append(item['Accuracy'])
        
        phases = list(phase_data.keys())
        phase_values = [phase_data[phase] for phase in phases]
        
        axes[0, 1].boxplot(phase_values, labels=phases)
        axes[0, 1].set_title('Accuracy by Phase')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Average accuracy over rounds
        round_accuracy = {}
        for agent_name, data in agent_data.items():
            round_accuracy[agent_name] = {}
            for item in data:
                round_num = item['Round']
                if round_num not in round_accuracy[agent_name]:
                    round_accuracy[agent_name][round_num] = []
                round_accuracy[agent_name][round_num].append(item['Accuracy'])
        
        for agent_name in agent_names:
            rounds = sorted(round_accuracy[agent_name].keys())
            avg_accuracies = [np.mean(round_accuracy[agent_name][r]) for r in rounds]
            axes[1, 0].plot(rounds, avg_accuracies, marker='o', label=agent_name)
        
        axes[1, 0].set_title('Average Accuracy Over Rounds')
        axes[1, 0].set_xlabel('Round Number')
        axes[1, 0].set_ylabel('Average Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Performance improvement
        for agent_name in agent_names:
            data = agent_data[agent_name]
            initial_acc = np.mean([item['Accuracy'] for item in data if item['Phase'] == 'initial'])
            # Get the final self-adjustment accuracy (last self-adjustment phase)
            self_adjustment_data = [item for item in data if 'Self-Adjustment' in item['Phase']]
            final_acc = np.mean([item['Accuracy'] for item in self_adjustment_data]) if self_adjustment_data else 0
            improvement = final_acc - initial_acc
            
            axes[1, 1].bar(agent_name, improvement, 
                          color='green' if improvement > 0 else 'red', alpha=0.7)
        
        axes[1, 1].set_title('Performance Improvement (Final - Initial)')
        axes[1, 1].set_ylabel('Accuracy Improvement')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_path, "agent_comparison.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Agent comparison saved to: {save_path}")
        return save_path
    
    def create_debate_flow_diagram(self, session: DebateSession,
                                 save_path: Optional[str] = None) -> str:
        """Create a flow diagram showing the debate process for a single session."""
        
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Define positions for agents
        agent_names = list(set([r.agent_name for r in session.initial_proposals]))
        agent_positions = {name: i for i, name in enumerate(agent_names)}
        
        # Plot initial proposals
        y_pos = 0
        ax.text(0, y_pos, "Initial Proposals", fontsize=14, fontweight='bold')
        y_pos -= 1
        
        for proposal in session.initial_proposals:
            x = agent_positions[proposal.agent_name]
            ax.scatter(x, y_pos, s=100, alpha=0.7)
            ax.text(x, y_pos-0.2, proposal.agent_name, ha='center', fontsize=10)
            y_pos -= 0.5
        
        # Plot debate rounds
        for round_data in session.debate_rounds:
            y_pos -= 1
            ax.text(0, y_pos, f"Round {round_data.round_number}: {round_data.player_name}", 
                   fontsize=12, fontweight='bold')
            y_pos -= 0.5
            
            # Plot debate responses
            for response in round_data.agent_responses:
                x = agent_positions[response.agent_name]
                ax.scatter(x, y_pos, s=80, alpha=0.6, color='orange')
                y_pos -= 0.3
            
            # Plot debate and self-adjustment responses
            y_pos -= 0.5
            for response in round_data.agent_responses:
                if response.phase in ['debate', 'self_adjustment']:
                    x = agent_positions[response.agent_name]
                    color = 'blue' if response.phase == 'debate' else 'green'
                    ax.scatter(x, y_pos, s=80, alpha=0.6, color=color)
                    y_pos -= 0.3
        
        # Plot final vote
        y_pos -= 1
        ax.text(0, y_pos, "Final Vote", fontsize=14, fontweight='bold')
        y_pos -= 0.5
        
        if session.final_vote:
            ax.scatter(0, y_pos, s=150, color='blue', alpha=0.8)
            ax.text(0, y_pos-0.3, "Majority Vote", ha='center', fontsize=10)
        
        if session.supervisor_decision:
            y_pos -= 1
            ax.scatter(0, y_pos, s=150, color='red', alpha=0.8)
            ax.text(0, y_pos-0.3, "Supervisor Decision", ha='center', fontsize=10)
        
        # Customize plot
        ax.set_xlim(-1, len(agent_names))
        ax.set_ylim(y_pos-1, 1)
        ax.set_xticks(range(len(agent_names)))
        ax.set_xticklabels(agent_names)
        ax.set_title(f'Debate Flow Diagram - Game {session.game_id}', fontsize=16)
        ax.grid(True, alpha=0.3)
        
        # Add legend
        legend_elements = [
            plt.scatter([], [], s=100, alpha=0.7, color='blue', label='Initial Proposal'),
            plt.scatter([], [], s=80, alpha=0.6, color='orange', label='Debate Response'),
            plt.scatter([], [], s=80, alpha=0.6, color='green', label='Self-Adjustment'),
            plt.scatter([], [], s=150, color='blue', alpha=0.8, label='Final Vote'),
            plt.scatter([], [], s=150, color='red', alpha=0.8, label='Supervisor Decision')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_path, f"debate_flow_{session.game_id}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"🔄 Debate flow diagram saved to: {save_path}")
        return save_path
    
    def create_summary_report(self, sessions: List[DebateSession],
                            save_path: Optional[str] = None) -> str:
        """Create a comprehensive summary report."""
        
        if save_path is None:
            save_path = os.path.join(self.output_path, "summary_report.txt")
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("MULTI-AGENT DEBATE SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total Sessions: {len(sessions)}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall statistics
            total_agents = len(set([r.agent_name for session in sessions for r in session.initial_proposals]))
            f.write(f"Total Agents: {total_agents}\n")
            
            # Calculate overall accuracy
            all_accuracies = []
            for session in sessions:
                for round_data in session.debate_rounds:
                    for response in round_data.agent_responses:
                        correct = 0
                        total = len(session.ground_truth_solution)
                        for player, role in session.ground_truth_solution.items():
                            if response.player_role_assignments.get(player) == role:
                                correct += 1
                        accuracy = correct / total if total > 0 else 0
                        all_accuracies.append(accuracy)
            
            if all_accuracies:
                f.write(f"Overall Average Accuracy: {np.mean(all_accuracies):.3f}\n")
                f.write(f"Accuracy Std Dev: {np.std(all_accuracies):.3f}\n")
                f.write(f"Best Accuracy: {np.max(all_accuracies):.3f}\n")
                f.write(f"Worst Accuracy: {np.min(all_accuracies):.3f}\n\n")
            
            # Per-agent statistics
            f.write("PER-AGENT STATISTICS:\n")
            f.write("-" * 30 + "\n")
            
            agent_stats = {}
            for session in sessions:
                # Include initial proposals
                for proposal in session.initial_proposals:
                    agent_name = proposal.agent_name
                    if agent_name not in agent_stats:
                        agent_stats[agent_name] = []
                    
                    correct = 0
                    total = len(session.ground_truth_solution)
                    for player, role in session.ground_truth_solution.items():
                        if proposal.player_role_assignments.get(player) == role:
                            correct += 1
                    accuracy = correct / total if total > 0 else 0
                    agent_stats[agent_name].append(accuracy)
                
                # Include debate rounds
                for round_data in session.debate_rounds:
                    for response in round_data.agent_responses:
                        agent_name = response.agent_name
                        if agent_name not in agent_stats:
                            agent_stats[agent_name] = []
                        
                        correct = 0
                        total = len(session.ground_truth_solution)
                        for player, role in session.ground_truth_solution.items():
                            if response.player_role_assignments.get(player) == role:
                                correct += 1
                        accuracy = correct / total if total > 0 else 0
                        agent_stats[agent_name].append(accuracy)
            
            for agent_name, accuracies in agent_stats.items():
                f.write(f"{agent_name}:\n")
                f.write(f"  Average Accuracy: {np.mean(accuracies):.3f}\n")
                f.write(f"  Std Dev: {np.std(accuracies):.3f}\n")
                f.write(f"  Total Responses: {len(accuracies)}\n\n")
            
            # Consensus statistics
            f.write("CONSENSUS STATISTICS:\n")
            f.write("-" * 30 + "\n")
            
            consensus_reached = 0
            supervisor_used = 0
            
            for session in sessions:
                if session.final_vote:
                    consensus_reached += 1
                if session.supervisor_decision:
                    supervisor_used += 1
            
            f.write(f"Sessions with Consensus: {consensus_reached}/{len(sessions)}\n")
            f.write(f"Sessions requiring Supervisor: {supervisor_used}/{len(sessions)}\n")
            f.write(f"Consensus Rate: {consensus_reached/len(sessions)*100:.1f}%\n")
        
        print(f"📋 Summary report saved to: {save_path}")
        return save_path
    
    def create_detailed_tracking_report(self, sessions: List[DebateSession],
                                       save_path: Optional[str] = None) -> str:
        """Create a detailed report showing how each agent's solution changes over rounds."""
        
        if save_path is None:
            save_path = os.path.join(self.output_path, "detailed_tracking.txt")
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("DETAILED AGENT SOLUTION TRACKING REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            for session in sessions:
                f.write(f"GAME {session.game_id}\n")
                f.write("-" * 20 + "\n")
                f.write(f"Ground Truth: {session.ground_truth_solution}\n\n")
                
                # Initial proposals
                f.write("INITIAL PROPOSALS:\n")
                f.write("-" * 20 + "\n")
                for proposal in session.initial_proposals:
                    f.write(f"{proposal.agent_name}:\n")
                    f.write(f"  Assignments: {proposal.player_role_assignments}\n")
                    
                    # Calculate accuracy for this proposal
                    correct = 0
                    total = len(session.ground_truth_solution)
                    for player, role in session.ground_truth_solution.items():
                        if proposal.player_role_assignments.get(player) == role:
                            correct += 1
                    accuracy = correct / total if total > 0 else 0
                    f.write(f"  Accuracy: {accuracy:.3f} ({correct}/{total})\n")
                    f.write(f"  Explanation: {proposal.explanation[:100]}...\n\n")
                
                # Debate rounds
                f.write("DEBATE ROUNDS:\n")
                f.write("-" * 20 + "\n")
                for round_data in session.debate_rounds:
                    f.write(f"Round {round_data.round_number} - Player: {round_data.player_name}\n")
                    f.write(f"Debate Summary: {round_data.debate_summary}\n")
                    
                    for response in round_data.agent_responses:
                        f.write(f"  {response.agent_name}:\n")
                        f.write(f"    Assignments: {response.player_role_assignments}\n")
                        
                        # Calculate accuracy for this response
                        correct = 0
                        total = len(session.ground_truth_solution)
                        for player, role in session.ground_truth_solution.items():
                            if response.player_role_assignments.get(player) == role:
                                correct += 1
                        accuracy = correct / total if total > 0 else 0
                        f.write(f"    Accuracy: {accuracy:.3f} ({correct}/{total})\n")
                        f.write(f"    Explanation: {response.explanation[:100]}...\n")
                    
                    f.write(f"  Consensus Reached: {round_data.consensus_reached}\n")
                    f.write(f"  Majority Role: {round_data.majority_role}\n\n")
                
                # Final results
                f.write("FINAL RESULTS:\n")
                f.write("-" * 20 + "\n")
                if session.final_vote:
                    f.write(f"Final Vote: {session.final_vote}\n")
                    # Calculate final accuracy
                    correct = 0
                    total = len(session.ground_truth_solution)
                    for player, role in session.ground_truth_solution.items():
                        if session.final_vote.get(player) == role:
                            correct += 1
                    accuracy = correct / total if total > 0 else 0
                    f.write(f"Final Accuracy: {accuracy:.3f} ({correct}/{total})\n")
                
                if session.supervisor_decision:
                    f.write(f"Supervisor Decision: {session.supervisor_decision}\n")
                    # Calculate supervisor accuracy
                    correct = 0
                    total = len(session.ground_truth_solution)
                    for player, role in session.ground_truth_solution.items():
                        if session.supervisor_decision.get(player) == role:
                            correct += 1
                    accuracy = correct / total if total > 0 else 0
                    f.write(f"Supervisor Accuracy: {accuracy:.3f} ({correct}/{total})\n")
                
                f.write("\n" + "=" * 50 + "\n\n")
        
        print(f"📋 Detailed tracking report saved to: {save_path}")
        return save_path
    
    def create_simple_detailed_per_player_accuracy(self, sessions: List[DebateSession], 
                                                 save_path: Optional[str] = None) -> str:
        """Create detailed visualization showing per-player accuracy by round for each agent (simplified for chat history)."""
        
        if not sessions:
            return ""
        
        # Collect detailed data from available session data
        detailed_data = []
        
        for session in sessions:
            players = list(session.ground_truth_solution.keys())
            
            # Initial proposals
            for proposal in session.initial_proposals:
                for player in players:
                    predicted_role = proposal.player_role_assignments.get(player, "unknown")
                    ground_truth_role = session.ground_truth_solution[player]
                    is_correct = predicted_role == ground_truth_role
                    
                    detailed_data.append({
                        'Game': session.game_id,
                        'Agent': proposal.agent_name,
                        'Player': player,
                        'Phase': 'Initial',
                        'Round': 0,
                        'Correct': is_correct,
                        'Predicted': predicted_role,
                        'GroundTruth': ground_truth_role
                    })
            
            # Debate rounds
            for round_data in session.debate_rounds:
                for response in round_data.agent_responses:
                    if response.phase == 'self_adjustment':
                        for player in players:
                            predicted_role = response.player_role_assignments.get(player, "unknown")
                            ground_truth_role = session.ground_truth_solution[player]
                            is_correct = predicted_role == ground_truth_role
                            
                            detailed_data.append({
                                'Game': session.game_id,
                                'Agent': response.agent_name,
                                'Player': player,
                                'Phase': f'After Round {round_data.round_number}',
                                'Round': round_data.round_number,
                                'Correct': is_correct,
                                'Predicted': predicted_role,
                                'GroundTruth': ground_truth_role
                            })
        
        if not detailed_data:
            return ""
        
        df = pd.DataFrame(detailed_data)
        
        # Create figure with subplots
        agents = df['Agent'].unique()
        players = df['Player'].unique()
        
        fig, axes = plt.subplots(len(agents), len(players), figsize=(4*len(players), 3*len(agents)))
        if len(agents) == 1:
            axes = axes.reshape(1, -1)
        if len(players) == 1:
            axes = axes.reshape(-1, 1)
        
        for i, agent in enumerate(agents):
            for j, player in enumerate(players):
                ax = axes[i, j]
                
                # Filter data for this agent and player
                agent_player_data = df[(df['Agent'] == agent) & (df['Player'] == player)]
                
                if len(agent_player_data) == 0:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{agent}\n{player}')
                    continue
                
                # Create timeline plot
                phases = agent_player_data['Phase'].tolist()
                correctness = agent_player_data['Correct'].tolist()
                
                # Plot correctness over phases
                colors = ['red' if not correct else 'green' for correct in correctness]
                bars = ax.bar(range(len(phases)), [1] * len(phases), color=colors, alpha=0.7)
                
                # Add phase labels
                ax.set_xticks(range(len(phases)))
                ax.set_xticklabels(phases, rotation=45, ha='right')
                ax.set_ylim(0, 1.2)
                
                # Add correctness labels
                for k, (bar, correct) in enumerate(zip(bars, correctness)):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                           '✓' if correct else '✗', ha='center', va='bottom', fontsize=12)
                
                ax.set_title(f'{agent}\n{player}')
                ax.set_ylabel('Correct')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = os.path.join(self.output_path, "detailed_per_player_accuracy.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"🔍 Simple detailed per-player accuracy saved to: {save_path}")
        return save_path
    
    def create_detailed_per_player_accuracy(self, sessions: List[DebateSession], 
                                          save_path: Optional[str] = None) -> str:
        """Create detailed visualization showing per-player accuracy by round for each agent."""
        
        if not sessions:
            return ""
        
        # Collect detailed data
        detailed_data = []
        
        for session in sessions:
            # Get detailed tracking data
            tracking = session.performance_tracking
            if "detailed_per_player_tracking" not in tracking:
                continue
                
            detailed_tracking = tracking["detailed_per_player_tracking"]
            players = list(session.ground_truth_solution.keys())
            
            # Initial proposals
            for agent_name, agent_data in detailed_tracking["initial"].items():
                for player in players:
                    if player in agent_data:
                        detailed_data.append({
                            'Game': session.game_id,
                            'Agent': agent_name,
                            'Player': player,
                            'Phase': 'Initial',
                            'Round': 0,
                            'Correct': agent_data[player]['correct'],
                            'Predicted': agent_data[player]['predicted'],
                            'GroundTruth': agent_data[player]['ground_truth']
                        })
            
            # Rounds
            for round_key, round_data in detailed_tracking["rounds"].items():
                round_num = int(round_key.split('_')[1])
                for agent_name, agent_data in round_data["agents"].items():
                    for player in players:
                        if player in agent_data:
                            detailed_data.append({
                                'Game': session.game_id,
                                'Agent': agent_name,
                                'Player': player,
                                'Phase': f'Round {round_num}',
                                'Round': round_num,
                                'Correct': agent_data[player]['correct'],
                                'Predicted': agent_data[player]['predicted'],
                                'GroundTruth': agent_data[player]['ground_truth']
                            })
            
            # Final vote
            if "majority_vote" in detailed_tracking["final"]:
                final_data = detailed_tracking["final"]["majority_vote"]
                for player in players:
                    if player in final_data:
                        detailed_data.append({
                            'Game': session.game_id,
                            'Agent': 'Majority Vote',
                            'Player': player,
                            'Phase': 'Final Vote',
                            'Round': 999,  # Use high number for final
                            'Correct': final_data[player]['correct'],
                            'Predicted': final_data[player]['predicted'],
                            'GroundTruth': final_data[player]['ground_truth']
                        })
            
            # Supervisor decision
            if "supervisor" in detailed_tracking["supervisor"]:
                supervisor_data = detailed_tracking["supervisor"]["supervisor"]
                for player in players:
                    if player in supervisor_data:
                        detailed_data.append({
                            'Game': session.game_id,
                            'Agent': 'Supervisor',
                            'Player': player,
                            'Phase': 'Supervisor',
                            'Round': 1000,  # Use higher number for supervisor
                            'Correct': supervisor_data[player]['correct'],
                            'Predicted': supervisor_data[player]['predicted'],
                            'GroundTruth': supervisor_data[player]['ground_truth']
                        })
        
        if not detailed_data:
            return ""
        
        df = pd.DataFrame(detailed_data)
        
        # Create figure with subplots
        agents = df['Agent'].unique()
        players = df['Player'].unique()
        
        fig, axes = plt.subplots(len(agents), len(players), figsize=(4*len(players), 3*len(agents)))
        if len(agents) == 1:
            axes = axes.reshape(1, -1)
        if len(players) == 1:
            axes = axes.reshape(-1, 1)
        
        for i, agent in enumerate(agents):
            for j, player in enumerate(players):
                ax = axes[i, j]
                
                # Filter data for this agent and player
                agent_player_data = df[(df['Agent'] == agent) & (df['Player'] == player)]
                
                if len(agent_player_data) == 0:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{agent}\n{player}')
                    continue
                
                # Create timeline plot
                rounds = sorted(agent_player_data['Round'].unique())
                correctness = []
                phases = []
                
                for round_num in rounds:
                    round_data = agent_player_data[agent_player_data['Round'] == round_num]
                    if len(round_data) > 0:
                        correctness.append(round_data.iloc[0]['Correct'])
                        phases.append(round_data.iloc[0]['Phase'])
                
                # Plot correctness over rounds
                colors = ['red' if not correct else 'green' for correct in correctness]
                bars = ax.bar(range(len(rounds)), [1] * len(rounds), color=colors, alpha=0.7)
                
                # Add phase labels
                ax.set_xticks(range(len(rounds)))
                ax.set_xticklabels(phases, rotation=45, ha='right')
                ax.set_ylim(0, 1.2)
                
                # Add correctness labels
                for k, (bar, correct) in enumerate(zip(bars, correctness)):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                           '✓' if correct else '✗', ha='center', va='bottom', fontsize=12)
                
                ax.set_title(f'{agent}\n{player}')
                ax.set_ylabel('Correct')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = os.path.join(self.output_path, "detailed_per_player_accuracy.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_per_player_prediction_matrix(self, sessions: List[DebateSession], 
                                          save_path: Optional[str] = None) -> str:
        """Create a fine-grained matrix showing each agent's prediction (correct/wrong) for each player at different checkpoints."""
        
        # Collect data for all phases
        data = []
        
        for session in sessions:
            gt_solution = session.ground_truth_solution
            
            # Initial proposals
            for proposal in session.initial_proposals:
                for player, gt_role in gt_solution.items():
                    predicted_role = proposal.player_role_assignments.get(player, "unknown")
                    is_correct = predicted_role == gt_role
                    
                    data.append({
                        'Game': session.game_id,
                        'Agent': proposal.agent_name,
                        'Player': player,
                        'Phase': 'Initial',
                        'Predicted': predicted_role,
                        'Ground_Truth': gt_role,
                        'Correct': is_correct
                    })
            
            # Debate phases for each round
            for round_data in session.debate_rounds:
                for response in round_data.agent_responses:
                    # Only include self-adjustment phases for the visualization
                    # This represents the state after each round's self-adjustment
                    if response.phase == 'self_adjustment':
                        for player, gt_role in gt_solution.items():
                            predicted_role = response.player_role_assignments.get(player, "unknown")
                            is_correct = predicted_role == gt_role
                            
                            data.append({
                                'Game': session.game_id,
                                'Agent': response.agent_name,
                                'Player': player,
                                'Phase': f'After Round {round_data.round_number}',
                                'Predicted': predicted_role,
                                'Ground_Truth': gt_role,
                                'Correct': is_correct
                            })
            
            # Final consensus (if available)
            if session.final_vote:
                for player, gt_role in gt_solution.items():
                    predicted_role = session.final_vote.get(player, "unknown")
                    is_correct = predicted_role == gt_role
                    
                    # Add final consensus for all agents (they should all have the same result)
                    for proposal in session.initial_proposals:
                        data.append({
                            'Game': session.game_id,
                            'Agent': proposal.agent_name,
                            'Player': player,
                            'Phase': 'Final Consensus',
                            'Predicted': predicted_role,
                            'Ground_Truth': gt_role,
                            'Correct': is_correct
                        })
        
        df = pd.DataFrame(data)
        
        # Create a pivot table showing correctness for each agent-player-phase combination
        pivot_table = df.pivot_table(
            values='Correct', 
            index=['Agent', 'Player'], 
            columns='Phase', 
            aggfunc='first'
        )
        
        # Convert boolean values to 1/0 for visualization
        pivot_table = pivot_table.astype(float)
        
        # Reorder columns for logical flow
        phase_columns = list(pivot_table.columns)
        initial_col = 'Initial'
        round_cols = [col for col in phase_columns if col.startswith('After Round')]
        final_consensus_col = 'Final Consensus' if 'Final Consensus' in phase_columns else None
        
        # Sort round columns by round number
        def extract_round_number(col):
            if col.startswith('After Round'):
                return int(col.split(' ')[2])
            return 0
        
        round_cols.sort(key=extract_round_number)
        
        # Build phase order: Initial -> Rounds -> Final Consensus
        phase_order = [initial_col] + round_cols
        if final_consensus_col:
            phase_order.append(final_consensus_col)
        
        pivot_table = pivot_table.reindex(columns=phase_order)
        
        # Create heatmap
        plt.figure(figsize=(max(10, len(phase_order) * 2), max(8, len(pivot_table) * 0.8)))
        
        # Create custom colormap (red for wrong, green for correct)
        from matplotlib.colors import LinearSegmentedColormap
        colors = ['#FF6B6B', '#4ECDC4']  # Red for 0 (wrong), Green for 1 (correct)
        correctness_cmap = LinearSegmentedColormap.from_list('correctness', colors, N=256)
        
        sns.heatmap(
            pivot_table, 
            annot=True, 
            cmap=correctness_cmap,
            cbar_kws={'label': 'Correct (1) / Wrong (0)'},
            fmt='.0f',
            vmin=0, vmax=1
        )
        
        plt.title('Per-Player Prediction Accuracy Matrix\n(1=Correct, 0=Wrong)', fontsize=14, fontweight='bold')
        plt.xlabel('Debate Phase', fontsize=12)
        plt.ylabel('Agent & Player', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save
        if save_path is None:
            save_path = os.path.join(self.output_path, "per_player_prediction_matrix.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_player_centric_analysis(self, sessions: List[DebateSession], 
                                      save_path: Optional[str] = None) -> str:
        """Create analysis showing initial vs final prediction accuracy for each player."""
        
        # Collect data
        data = []
        
        for session in sessions:
            gt_solution = session.ground_truth_solution
            
            # Get initial predictions for each agent
            initial_predictions = {}
            for proposal in session.initial_proposals:
                initial_predictions[proposal.agent_name] = proposal.player_role_assignments
            
            # Get final predictions (consensus or supervisor decision)
            final_predictions = session.final_vote if session.final_vote else session.supervisor_decision
            
            # For each player, calculate initial and final accuracy
            for player, gt_role in gt_solution.items():
                # Calculate initial accuracy (how many agents got it right initially)
                initial_correct = 0
                total_agents = len(initial_predictions)
                
                for agent_name, predictions in initial_predictions.items():
                    if predictions.get(player) == gt_role:
                        initial_correct += 1
                
                initial_accuracy = initial_correct / total_agents if total_agents > 0 else 0
                
                # Calculate final accuracy (whether final consensus got it right)
                final_correct = 1 if final_predictions and final_predictions.get(player) == gt_role else 0
                
                data.append({
                    'Game': session.game_id,
                    'Player': player,
                    'Ground_Truth': gt_role,
                    'Initial_Accuracy': initial_accuracy,
                    'Final_Correct': final_correct,
                    'Improvement': final_correct - initial_accuracy
                })
        
        df = pd.DataFrame(data)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: Initial vs Final accuracy by player
        players = df['Player'].unique()
        x_pos = np.arange(len(players))
        
        initial_acc = [df[df['Player'] == p]['Initial_Accuracy'].mean() for p in players]
        final_acc = [df[df['Player'] == p]['Final_Correct'].mean() for p in players]
        
        width = 0.35
        ax1.bar(x_pos - width/2, initial_acc, width, label='Initial Accuracy', alpha=0.8, color='#FF6B6B')
        ax1.bar(x_pos + width/2, final_acc, width, label='Final Correct', alpha=0.8, color='#4ECDC4')
        
        ax1.set_xlabel('Player')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Initial vs Final Prediction Accuracy by Player')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(players)
        ax1.legend()
        ax1.set_ylim(0, 1.1)
        
        # Add value labels on bars
        for i, (init, final) in enumerate(zip(initial_acc, final_acc)):
            ax1.text(i - width/2, init + 0.02, f'{init:.2f}', ha='center', va='bottom')
            ax1.text(i + width/2, final + 0.02, f'{final:.2f}', ha='center', va='bottom')
        
        # Plot 2: Improvement analysis
        improvement_data = df.groupby('Player')['Improvement'].mean()
        colors = ['#4ECDC4' if x >= 0 else '#FF6B6B' for x in improvement_data.values]
        
        bars = ax2.bar(improvement_data.index, improvement_data.values, color=colors, alpha=0.8)
        ax2.set_xlabel('Player')
        ax2.set_ylabel('Improvement (Final - Initial)')
        ax2.set_title('Prediction Improvement by Player')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_ylim(-1.1, 1.1)
        
        # Add value labels on bars
        for bar, value in zip(bars, improvement_data.values):
            ax2.text(bar.get_x() + bar.get_width()/2, value + (0.05 if value >= 0 else -0.05), 
                    f'{value:.2f}', ha='center', va='bottom' if value >= 0 else 'top')
        
        plt.tight_layout()
        
        # Save
        if save_path is None:
            save_path = os.path.join(self.output_path, "player_centric_analysis.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_all_visualizations(self, sessions: List[DebateSession]) -> Dict[str, str]:
        """Create all visualizations and return file paths."""
        results = {}
        
        print("🎨 Creating visualizations...")
        
        # Performance matrix (overall accuracy)
        results['performance_matrix'] = self.create_performance_matrix(sessions)
        
        # Fine-grained per-player prediction matrix
        results['per_player_prediction_matrix'] = self.create_per_player_prediction_matrix(sessions)
        
        # Player-centric analysis: Initial vs Final
        results['player_centric_analysis'] = self.create_player_centric_analysis(sessions)
        
        # Consensus tracking
        results['consensus_tracking'] = self.create_consensus_tracking(sessions)
        
        # Agent comparison
        results['agent_comparison'] = self.create_agent_comparison(sessions)
        
        # Detailed per-player accuracy
        results['detailed_per_player_accuracy'] = self.create_detailed_per_player_accuracy(sessions)
        
        # Summary report
        results['summary_report'] = self.create_summary_report(sessions)
        
        # Detailed tracking report
        results['detailed_tracking'] = self.create_detailed_tracking_report(sessions)
        
        print(f"✅ All visualizations created in: {self.output_path}")
        return results
    
    def create_all_visualizations_from_folder(self) -> Dict[str, str]:
        """Create all visualizations by loading JSON files from the current folder."""
        # Find all JSON files in the current folder
        json_files = [f for f in os.listdir(self.output_path) if f.endswith('.json')]
        
        if not json_files:
            print(f"❌ No JSON files found in {self.output_path}")
            return {}
        
        print(f"📊 Found {len(json_files)} JSON files")
        
        # Load sessions from JSON files
        sessions = []
        for json_file in json_files:
            try:
                json_path = os.path.join(self.output_path, json_file)
                with open(json_path, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                
                # Convert back to DebateSession object
                session = self._json_to_debate_session(session_data)
                sessions.append(session)
                print(f"✅ Loaded {json_file}")
                
            except Exception as e:
                print(f"❌ Error loading {json_file}: {e}")
                continue
        
        if not sessions:
            print(f"❌ No valid sessions loaded")
            return {}
        
        print(f"🎯 Creating visualizations for {len(sessions)} sessions")
        return self.create_all_visualizations(sessions)
    
    def _json_to_debate_session(self, session_data: Dict[str, Any]) -> DebateSession:
        """Convert JSON data back to DebateSession object."""
        # Use the more complete conversion function
        return _dict_to_debate_session(session_data)

def load_debate_sessions(results_dir: str) -> List[DebateSession]:
    """Load debates from JSON files in the results directory and all subdirectories."""
    sessions = []
    
    # Find all debate JSON files recursively
    for root, dirs, files in os.walk(results_dir):
        for filename in files:
            if filename.startswith('debate_') and filename.endswith('.json'):
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        session_data = json.load(f)
                    
                    # Convert back to DebateSession object
                    session = _dict_to_debate_session(session_data)
                    # Store the source file path for later use
                    session.source_file = filepath
                    sessions.append(session)
                    print(f"✅ Loaded debate from {os.path.relpath(filepath, results_dir)}")
                    
                except Exception as e:
                    print(f"❌ Error loading {os.path.relpath(filepath, results_dir)}: {e}")
    
    return sessions

def _dict_to_debate_session(data: Dict[str, Any]) -> DebateSession:
    """Convert dictionary back to DebateSession object."""
    
    # Convert initial proposals
    initial_proposals = []
    for proposal_data in data.get('initial_proposals', []):
        proposal = AgentResponse(
            agent_name=proposal_data['agent_name'],
            game_id=proposal_data['game_id'],
            round_number=proposal_data['round_number'],
            phase=proposal_data['phase'],
            player_role_assignments=proposal_data['player_role_assignments'],
            explanation=proposal_data['explanation'],
            confidence=proposal_data.get('confidence', 0.0),
            response_obj=proposal_data.get('response_obj'),
            timestamp=proposal_data.get('timestamp', ''),
            error=proposal_data.get('error', ''),
            agree_with=proposal_data.get('agree_with'),
            disagree_with=proposal_data.get('disagree_with'),
            agree_reasoning=proposal_data.get('agree_reasoning'),
            disagree_reasoning=proposal_data.get('disagree_reasoning')
        )
        initial_proposals.append(proposal)
    
    # Convert debate rounds
    debate_rounds = []
    for round_data in data.get('debate_rounds', []):
        agent_responses = []
        for response_data in round_data['agent_responses']:
            response = AgentResponse(
                agent_name=response_data['agent_name'],
                game_id=response_data['game_id'],
                round_number=response_data['round_number'],
                phase=response_data['phase'],
                player_role_assignments=response_data['player_role_assignments'],
                explanation=response_data['explanation'],
                confidence=response_data.get('confidence', 0.0),
                response_obj=response_data.get('response_obj'),
                timestamp=response_data.get('timestamp', ''),
                error=response_data.get('error', ''),
                agree_with=response_data.get('agree_with'),
                disagree_with=response_data.get('disagree_with'),
                agree_reasoning=response_data.get('agree_reasoning'),
                disagree_reasoning=response_data.get('disagree_reasoning')
            )
            agent_responses.append(response)
        
        debate_round = DebateRound(
            player_name=round_data['player_name'],
            round_number=round_data['round_number'],
            agent_responses=agent_responses,
            debate_summary=round_data.get('debate_summary', ''),
            consensus_reached=round_data.get('consensus_reached', False),
            majority_role=round_data.get('majority_role')
        )
        debate_rounds.append(debate_round)
    
    # Create DebateSession
    session = DebateSession(
        game_id=data['game_id'],
        game_text=data['game_text'],
        ground_truth_solution=data['ground_truth_solution'],
        initial_proposals=initial_proposals,
        debate_rounds=debate_rounds,
        final_vote=data.get('final_vote'),
        supervisor_decision=data.get('supervisor_decision'),
        performance_tracking=data.get('performance_tracking')
    )
    
    return session

def main():
    """Main function for standalone visualization."""
    parser = argparse.ArgumentParser(
        description="Generate visualizations from existing debate results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python debate_visualizer.py --results-dir ./debate_results
  python debate_visualizer.py --results-dir ./debate_results --output-dir ./visualizations
  python debate_visualizer.py --results-dir ./debate_results --specific-session debate_session_1_20250910_133709.json
        """
    )
    
    parser.add_argument(
        '--results-dir', '-r',
        type=str,
        default='./debate_results',
        help='Directory containing debate session JSON files (default: ./debate_results)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Output directory for visualizations (default: same as results-dir)'
    )
    
    parser.add_argument(
        '--specific-session', '-s',
        type=str,
        default=None,
        help='Generate visualizations for a specific debate file only'
    )
    
    parser.add_argument(
        '--no-performance-matrix',
        action='store_true',
        help='Skip performance matrix generation'
    )
    
    parser.add_argument(
        '--no-consensus-tracking',
        action='store_true',
        help='Skip consensus tracking visualization'
    )
    
    parser.add_argument(
        '--no-agent-comparison',
        action='store_true',
        help='Skip agent comparison visualization'
    )
    
    args = parser.parse_args()
    
    # Validate results directory
    if not os.path.exists(args.results_dir):
        print(f"❌ Results directory not found: {args.results_dir}")
        return
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = args.results_dir
    
    print(f"🔍 Loading debates from: {args.results_dir}")
    
    # Load debates
    if args.specific_session:
        # Load specific debate
        filepath = os.path.join(args.results_dir, args.specific_session)
        if not os.path.exists(filepath):
            print(f"❌ Specific debate file not found: {filepath}")
            return
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            session = _dict_to_debate_session(session_data)
            sessions = [session]
            print(f"✅ Loaded specific debate: {args.specific_session}")
        except Exception as e:
            print(f"❌ Error loading specific debate: {e}")
            return
    else:
        # Load all debates
        sessions = load_debate_sessions(args.results_dir)
    
    if not sessions:
        print("❌ No debates found!")
        return
    
    print(f"📊 Found {len(sessions)} debate(s)")
    
    # Generate visualizations for each session individually
    print("🎨 Generating visualizations...")
    results = {}
    
    for i, session in enumerate(sessions, 1):
        print(f"  📊 Processing session {i}/{len(sessions)} (Game ID: {session.game_id})")
        
        # Determine the output directory for this session (same as JSON file location)
        session_output_dir = os.path.dirname(session.source_file) if hasattr(session, 'source_file') else args.output_dir
        
        # Create visualizer for this session
        visualizer = ChatHistoryDebateVisualizer(session_output_dir)
        
        # Generate visualizations for this single session
        if not args.no_performance_matrix:
            print(f"    📈 Creating performance matrix...")
            results[f'performance_matrix_{session.game_id}'] = visualizer.create_performance_matrix([session])
        
        # Fine-grained per-player prediction matrix
        print(f"    🎯 Creating per-player prediction matrix...")
        results[f'per_player_prediction_matrix_{session.game_id}'] = visualizer.create_per_player_prediction_matrix([session])
        
        # Player-centric analysis: Initial vs Final
        print(f"    📊 Creating player-centric analysis...")
        results[f'player_centric_analysis_{session.game_id}'] = visualizer.create_player_centric_analysis([session])
        
        if not args.no_consensus_tracking:
            print(f"    📊 Creating consensus tracking...")
            results[f'consensus_tracking_{session.game_id}'] = visualizer.create_consensus_tracking([session])
        
        if not args.no_agent_comparison:
            print(f"    🤖 Creating agent comparison...")
            results[f'agent_comparison_{session.game_id}'] = visualizer.create_agent_comparison([session])
        
        # Detailed per-player accuracy (simplified version for chat history)
        print(f"    🔍 Creating detailed per-player accuracy...")
        results[f'detailed_per_player_accuracy_{session.game_id}'] = visualizer.create_simple_detailed_per_player_accuracy([session])
    
    print(f"\n✅ All visualizations completed!")
    print(f"📁 Results saved in individual session directories")
    print(f"\nGenerated files:")
    for name, path in results.items():
        print(f"  - {name}: {os.path.basename(path)}")

if __name__ == "__main__":
    main()
