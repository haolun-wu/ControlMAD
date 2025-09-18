#!/usr/bin/env python3
"""
Script to create scatter plots from record.md
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def parse_record_file(record_file):
    """Parse record.md and extract model data"""
    if not os.path.exists(record_file):
        print(f"❌ Record file not found: {record_file}")
        return None
    
    with open(record_file, 'r') as f:
        content = f.read()
    
    # Parse models
    models_data = {}
    lines = content.split('\n')
    
    current_model = None
    current_data = {}
    
    for line in lines:
        line = line.strip()
        if line.startswith('## '):
            # Save previous model if exists
            if current_model and current_data:
                models_data[current_model] = current_data.copy()
            
            # Start new model
            current_model = line[3:].strip()
            current_data = {}
        elif line.startswith('- Accuracy:'):
            try:
                # Parse accuracy in format "X/50"
                accuracy_str = line.split(':')[1].strip()
                accuracy_parts = accuracy_str.split('/')
                if len(accuracy_parts) == 2:
                    correct = float(accuracy_parts[0])
                    total = float(accuracy_parts[1])
                    current_data['accuracy'] = (correct / total) * 100  # Convert to percentage
            except:
                pass
        elif line.startswith('- Confidence:'):
            try:
                # Parse confidence in format "X/10" or "X/100"
                confidence_str = line.split(':')[1].strip()
                confidence_parts = confidence_str.split('/')
                if len(confidence_parts) == 2:
                    confidence_val = float(confidence_parts[0])
                    confidence_max = float(confidence_parts[1])
                    # Normalize to 1-10 scale
                    current_data['confidence'] = (confidence_val / confidence_max) * 10
            except:
                pass
    
    # Save last model
    if current_model and current_data:
        models_data[current_model] = current_data
    
    return models_data

def create_scatter_plot(models_data):
    """Create scatter plot for confidence vs accuracy with improved visualization and groupings"""
    if not models_data:
        print("❌ No model data found to plot")
        return
    
    # Filter models with complete data
    complete_models = {}
    for model_name, data in models_data.items():
        if all(key in data for key in ['accuracy', 'confidence']):
            complete_models[model_name] = data
    
    if not complete_models:
        print("❌ No models with complete data found")
        return
    
    # Extract data
    models = list(complete_models.keys())
    accuracy = [complete_models[model]['accuracy'] for model in models]
    confidence = [complete_models[model]['confidence'] for model in models]
    
    # Define grouping thresholds
    # Accuracy groups: High (>=70%), Medium (30-69%), Low (<30%)
    # Confidence groups: High (8.5-10), Medium (6.5-8.5), Low (1-6.5)
    accuracy_high_threshold = 70
    accuracy_medium_threshold = 30
    confidence_high_threshold = 8.5
    confidence_medium_threshold = 6.5
    
    # Group models
    accuracy_groups = {'High': [], 'Medium': [], 'Low': []}
    confidence_groups = {'High': [], 'Medium': [], 'Low': []}
    
    for i, model in enumerate(models):
        acc = accuracy[i]
        conf = confidence[i]
        
        # Group by accuracy
        if acc >= accuracy_high_threshold:
            accuracy_groups['High'].append(model)
        elif acc >= accuracy_medium_threshold:
            accuracy_groups['Medium'].append(model)
        else:
            accuracy_groups['Low'].append(model)
        
        # Group by confidence
        if conf >= confidence_high_threshold:
            confidence_groups['High'].append(model)
        elif conf >= confidence_medium_threshold:
            confidence_groups['Medium'].append(model)
        else:
            confidence_groups['Low'].append(model)
    
    # Print groupings
    print(f"📊 Creating plot for {len(models)} models:")
    print("\n🎯 Accuracy Groups:")
    for group, model_list in accuracy_groups.items():
        print(f"   {group}: {len(model_list)} models")
        for model in model_list:
            acc = complete_models[model]['accuracy']
            conf = complete_models[model]['confidence']
            short_name = model.replace('gpt-5-nano, ', '').replace(' (max output token = None, thinking budget = 4096)', '').replace(' (max output token = 4096, thinking_type = enabled, not able to limit the thinking effort)', '')
            print(f"     - {short_name}: {acc:.1f}% accuracy, {conf:.1f}/10 confidence")
    
    print("\n🎯 Confidence Groups:")
    for group, model_list in confidence_groups.items():
        print(f"   {group}: {len(model_list)} models")
        for model in model_list:
            acc = complete_models[model]['accuracy']
            conf = complete_models[model]['confidence']
            short_name = model.replace('gpt-5-nano, ', '').replace(' (max output token = None, thinking budget = 4096)', '').replace(' (max output token = 4096, thinking_type = enabled, not able to limit the thinking effort)', '')
            print(f"     - {short_name}: {acc:.1f}% accuracy, {conf:.1f}/10 confidence")
    
    # Create figure with improved layout
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Color code points by accuracy group
    colors = []
    for i, model in enumerate(models):
        acc = accuracy[i]
        if acc >= accuracy_high_threshold:
            colors.append('green')  # High accuracy
        elif acc >= accuracy_medium_threshold:
            colors.append('orange')  # Medium accuracy
        else:
            colors.append('red')  # Low accuracy
    
    # Create scatter plot with color coding
    scatter = ax.scatter(confidence, accuracy, alpha=0.8, s=150, c=colors, edgecolors='black', linewidth=1.5)
    
    # Set labels and title
    ax.set_xlabel('Confidence (1-10 scale)', fontsize=16)
    ax.set_ylabel('Accuracy (%)', fontsize=16)
    ax.set_title('Model Performance: Confidence vs Accuracy\n(Color-coded by Accuracy Level)', fontsize=18, fontweight='bold')
    
    # Improved x-axis with margins to prevent points on edges
    ax.set_xlim(-0.5, 10.5)
    ax.set_xticks(np.arange(0, 11, 1))  # Show all integer values
    ax.set_xticklabels([f'{i}' for i in range(11)])
    
    # Set y-axis with margins
    ax.set_ylim(-5, 105)
    ax.set_yticks(np.arange(0, 101, 10))
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Add grouping lines
    ax.axhline(y=accuracy_high_threshold, color='green', linestyle='--', alpha=0.5, linewidth=2)
    ax.axhline(y=accuracy_medium_threshold, color='orange', linestyle='--', alpha=0.5, linewidth=2)
    ax.axvline(x=confidence_high_threshold, color='blue', linestyle='--', alpha=0.5, linewidth=2)
    ax.axvline(x=confidence_medium_threshold, color='purple', linestyle='--', alpha=0.5, linewidth=2)
    
    # Add model labels with smart positioning to avoid overlaps
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.patches as mpatches
    
    # Create a list to track used positions
    used_positions = []
    
    for i, model in enumerate(models):
        short_name = model.replace('gpt-5-nano, ', '').replace(' (max output token = None, thinking budget = 4096)', '').replace(' (max output token = 4096, thinking_type = enabled, not able to limit the thinking effort)', '')
        
        # Smart positioning to avoid overlaps
        x_offset = 12
        y_offset = 12
        
        # Adjust offset based on position to avoid overlaps
        if confidence[i] > 8:  # High confidence - move left
            x_offset = -15
        elif confidence[i] < 6:  # Low confidence - move right
            x_offset = 15
        
        if accuracy[i] > 80:  # High accuracy - move down
            y_offset = -15
        elif accuracy[i] < 20:  # Low accuracy - move up
            y_offset = 15
        
        ax.annotate(short_name, (confidence[i], accuracy[i]), 
                    xytext=(x_offset, y_offset), textcoords='offset points', 
                    fontsize=9, alpha=0.9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=0.5))
    
    # Add legend for color coding
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', label=f'High Accuracy (≥{accuracy_high_threshold}%)'),
                      Patch(facecolor='orange', label=f'Medium Accuracy ({accuracy_medium_threshold}-{accuracy_high_threshold-1}%)'),
                      Patch(facecolor='red', label=f'Low Accuracy (<{accuracy_medium_threshold}%)')]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=12)
    
    # Add text annotations for grouping thresholds in a single row at the top
    # Position them at the middle of their corresponding confidence ranges
    ax.text((0 + confidence_medium_threshold)/2, 100, 'Low Confidence\n(1-6.5)', ha='center', va='center', 
            fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))
    ax.text((confidence_medium_threshold + confidence_high_threshold)/2, 100, 'Medium Confidence\n(6.5-8.5)', ha='center', va='center', 
            fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    ax.text((confidence_high_threshold + 10)/2, 100, 'High Confidence\n(8.5-10)', ha='center', va='center', 
            fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(project_root, "baseline", "confidence_vs_accuracy.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\n📊 Enhanced scatter plot saved to: {plot_file}")
    
    # Show plot
    plt.show()
    
    return accuracy_groups, confidence_groups

def main():
    """Main function"""
    record_file = os.path.join(project_root, "baseline", "record.md")
    
    print("📊 Model Performance Visualization")
    print("="*50)
    print(f"📁 Reading from: {record_file}")
    
    # Parse record file
    models_data = parse_record_file(record_file)
    
    if models_data:
        print(f"✅ Found {len(models_data)} models in record")
        
        # Create scatter plot and get groupings
        accuracy_groups, confidence_groups = create_scatter_plot(models_data)
        
        # Print summary
        print("\n📋 SUMMARY:")
        print("="*50)
        print("🎯 Accuracy Distribution:")
        for group, models in accuracy_groups.items():
            print(f"   {group}: {len(models)} models")
        
        print("\n🎯 Confidence Distribution:")
        for group, models in confidence_groups.items():
            print(f"   {group}: {len(models)} models")
    else:
        print("❌ No data found to plot")

if __name__ == "__main__":
    main()
