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
    
    # Handle overlapping points by adding small jitter
    import numpy as np
    from collections import defaultdict
    
    # Group models by their exact coordinates
    coordinate_groups = defaultdict(list)
    for i, model in enumerate(models):
        coord_key = (round(confidence[i], 1), round(accuracy[i], 1))
        coordinate_groups[coord_key].append(i)
    
    # Apply jitter to overlapping points
    jitter_amount = 0.15  # Small offset amount
    for coord_key, model_indices in coordinate_groups.items():
        if len(model_indices) > 1:  # Multiple models at same position
            print(f"⚠️  Found {len(model_indices)} overlapping models at confidence={coord_key[0]}, accuracy={coord_key[1]}%:")
            for idx in model_indices:
                print(f"     - {models[idx]}")
            
            # Apply circular jitter pattern for overlapping points
            for j, idx in enumerate(model_indices):
                if j > 0:  # Keep first point at original position
                    angle = 2 * np.pi * j / len(model_indices)
                    confidence[idx] += jitter_amount * np.cos(angle)
                    accuracy[idx] += jitter_amount * 2 * np.sin(angle)  # Slightly larger jitter for accuracy
    
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
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
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
    
    # Set labels (removed title)
    ax.set_xlabel('Confidence (1-10 scale)', fontsize=16)
    ax.set_ylabel('Accuracy (%)', fontsize=16)
    
    # Improved x-axis with margins to prevent points on edges
    ax.set_xlim(-0.5, 10.5)
    ax.set_xticks(np.arange(0, 11, 1))  # Show all integer values
    ax.set_xticklabels([f'{i}' for i in range(11)])
    
    # Set y-axis with margins - plot area goes to 100, extra space for boxes above
    ax.set_ylim(-5, 100)
    ax.set_yticks(np.arange(0, 101, 10))
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Add grouping lines
    ax.axhline(y=accuracy_high_threshold, color='green', linestyle='--', alpha=0.5, linewidth=2)
    ax.axhline(y=accuracy_medium_threshold, color='orange', linestyle='--', alpha=0.5, linewidth=2)
    ax.axvline(x=confidence_high_threshold, color='blue', linestyle='--', alpha=0.5, linewidth=2)
    ax.axvline(x=confidence_medium_threshold, color='purple', linestyle='--', alpha=0.5, linewidth=2)
    
    # Add model labels with enhanced positioning to avoid overlaps
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.patches as mpatches
    
    # Create a list to track used positions and avoid overlaps
    used_positions = []
    label_spacing = 35  # Increased minimum distance between labels
    
    def find_best_position(x, y, used_positions, label_spacing=35):
        """Find the best position for a label to avoid overlaps"""
        # Expanded offset options with better spacing and more variety
        offset_options = [
            # Close positions (preferred)
            (25, 25), (-30, 25), (25, -30), (-30, -30),   # Diagonal close
            (40, 0), (-45, 0), (0, 40), (0, -45),         # Cardinal close
            
            # Medium distance positions
            (50, 20), (-55, 20), (50, -25), (-55, -25),   # Medium diagonal
            (35, 35), (-40, 35), (35, -40), (-40, -40),   # Medium diagonal 2
            (60, 0), (-65, 0), (0, 60), (0, -65),         # Medium cardinal
            
            # Far positions for crowded areas
            (70, 15), (-75, 15), (70, -20), (-75, -20),   # Far horizontal
            (20, 70), (-25, 70), (20, -75), (-25, -75),   # Far vertical
            (55, 45), (-60, 45), (55, -50), (-60, -50),   # Far diagonal
            
            # Very far positions as last resort
            (85, 10), (-90, 10), (85, -15), (-90, -15),   # Very far horizontal
            (15, 85), (-20, 85), (15, -90), (-20, -90),   # Very far vertical
        ]
        
        for x_offset, y_offset in offset_options:
            # Calculate the actual position in data coordinates
            label_x = x + x_offset * 0.08  # Slightly smaller conversion factor
            label_y = y + y_offset * 0.4   # Slightly smaller conversion factor
            
            # Check if this position conflicts with existing labels
            conflict = False
            for used_x, used_y in used_positions:
                distance = ((label_x - used_x)**2 + (label_y - used_y)**2)**0.5
                if distance < label_spacing:
                    conflict = True
                    break
            
            if not conflict:
                return x_offset, y_offset, label_x, label_y
        
        # If no good position found, use spiral pattern
        import math
        spiral_radius = 50
        spiral_angle = len(used_positions) * 0.618 * 2 * math.pi  # Golden angle
        x_offset = spiral_radius * math.cos(spiral_angle)
        y_offset = spiral_radius * math.sin(spiral_angle)
        label_x = x + x_offset * 0.08
        label_y = y + y_offset * 0.4
        return x_offset, y_offset, label_x, label_y
    
    # Manual positioning for specific models to avoid overlaps
    manual_positions = {
        'o4-mini': (-45, 25),  
        'o3-mini': (25, -25), 
        'qwq-32b': (10, 10), 
        'gpt-5-mini (Medium)': (10, -20), 
        'gpt-5-mini (Low)': (-65, -25),  
        'doubao-seed-1-6-250615': (-45, 15),  
        'doubao-seed-1-6-flash-250828': (15, -10),  
        'gpt-5-nano-medium-effort': (35, -30), 
        'gemini-2.5-flash': (25, -25),  
        'qwen-plus': (-55, -10), 
        'qwen-turbo-latest': (25, -10), 
        'qwen-flash': (35, -30),  
        'gpt-4o-mini': (25, -25),  
    }
    
    for i, model in enumerate(models):
        short_name = model.replace('gpt-5-nano, ', '').replace(' (max output token = None, thinking budget = 4096)', '').replace(' (max output token = 4096, thinking_type = enabled, not able to limit the thinking effort)', '')
        
        # Clean up model names but keep them readable
        short_name = short_name.replace('gpt-5-mini-low-effort', 'gpt-5-mini (Low)')
        short_name = short_name.replace('gpt-5-mini-medium-effort', 'gpt-5-mini (Medium)')
        short_name = short_name.replace('gpt-5-mini-high-effort', 'gpt-5-mini (High)')
        short_name = short_name.replace('gpt-5-nano-low-effort', 'gpt-5-nano (Low)')
        short_name = short_name.replace('gpt-5-nano-medium-effort', 'gpt-5-nano (Medium)')
        short_name = short_name.replace('gpt-5-nano-high-effort', 'gpt-5-nano (High)')
        # Clean up other effort indicators
        short_name = short_name.replace(' (low effort)', ' (Low)')
        short_name = short_name.replace(' (medium effort)', ' (Medium)')
        short_name = short_name.replace(' (high effort)', ' (High)')
        short_name = short_name.replace('-low-effort', ' (Low)')
        short_name = short_name.replace('-medium-effort', ' (Medium)')
        short_name = short_name.replace('-high-effort', ' (High)')
        
        # Check if this model has manual positioning
        if short_name in manual_positions or model in manual_positions:
            # Use manual position
            position_key = short_name if short_name in manual_positions else model
            x_offset, y_offset = manual_positions[position_key]
            label_x = confidence[i] + x_offset * 0.08
            label_y = accuracy[i] + y_offset * 0.4
        else:
            # Use automatic positioning
            x_offset, y_offset, label_x, label_y = find_best_position(
                confidence[i], accuracy[i], used_positions, label_spacing
            )
        
        # Add this position to used positions
        used_positions.append((label_x, label_y))
        
        ax.annotate(short_name, (confidence[i], accuracy[i]), 
                    xytext=(x_offset, y_offset), textcoords='offset points', 
                    fontsize=7, alpha=0.95, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.95, edgecolor='black', linewidth=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.05', color='darkgray', alpha=0.8, lw=1.0))
    
    # Add legend for color coding
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', label=f'High Accuracy (≥{accuracy_high_threshold}%)'),
                      Patch(facecolor='orange', label=f'Medium Accuracy ({accuracy_medium_threshold}-{accuracy_high_threshold-1}%)'),
                      Patch(facecolor='red', label=f'Low Accuracy (<{accuracy_medium_threshold}%)')]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=12)
    
    # Add text annotations for grouping thresholds above the plot boundary
    # Position boxes in the horizontal center of each confidence zone
    low_confidence_center = (0 + confidence_medium_threshold) / 2  # Center of 0 to 6.5
    medium_confidence_center = (confidence_medium_threshold + confidence_high_threshold) / 2  # Center of 6.5 to 8.5
    high_confidence_center = (confidence_high_threshold + 10) / 2  # Center of 8.5 to 10
    
    ax.text(low_confidence_center, 1.08, 'Low Confidence\n(1-6.5)', ha='center', va='center', 
            fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8),
            transform=ax.get_xaxis_transform())
    ax.text(medium_confidence_center, 1.08, 'Medium Confidence\n(6.5-8.5)', ha='center', va='center', 
            fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8),
            transform=ax.get_xaxis_transform())
    ax.text(high_confidence_center, 1.08, 'High Confidence\n(8.5-10)', ha='center', va='center', 
            fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8),
            transform=ax.get_xaxis_transform())
    
    plt.tight_layout()
    
    # Save plot in both PNG and PDF formats
    plot_file_png = os.path.join(project_root, "baseline", "model_summary.png")
    plot_file_pdf = os.path.join(project_root, "baseline", "model_summary.pdf")
    
    plt.savefig(plot_file_png, dpi=300, bbox_inches='tight')
    plt.savefig(plot_file_pdf, format='pdf', bbox_inches='tight')
    
    print(f"\n📊 Enhanced scatter plot saved to:")
    print(f"   📄 PNG: {plot_file_png}")
    print(f"   📄 PDF: {plot_file_pdf}")
    
    # Show plot (comment out if running headless)
    plt.show()
    print("📊 Plot generation completed!")
    
    return accuracy_groups, confidence_groups

def main():
    """Main function"""
    record_file = os.path.join(project_root, "baseline", "record.md")
    
    print("📊 Model Performance Visualization")
    print("="*50)
    print(f"📁 Reading from: {record_file}")
    print(f"📁 File exists: {os.path.exists(record_file)}")
    
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
