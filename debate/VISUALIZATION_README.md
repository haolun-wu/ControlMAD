# Debate Results Visualization

This directory contains tools for visualizing debate results independently from running the debate system.

## Standalone Visualization Script

### `debate_visualizer.py`

Generate visualizations from existing debate session JSON files without needing to run the debate system again.

#### Usage:

```bash
# Generate all visualizations from debate_results directory
python debate/debate_visualizer.py --results-dir ./debate_results

# Generate visualizations for a specific session
python debate/debate_visualizer.py --results-dir ./debate_results --specific-session debate_session_1_20250910_133709.json

# Generate visualizations with custom output directory
python debate/debate_visualizer.py --results-dir ./debate_results --output-dir ./my_visualizations

# Skip certain visualizations
python debate/debate_visualizer.py --results-dir ./debate_results --no-performance-matrix --no-consensus-tracking
```

#### Options:

- `--results-dir, -r`: Directory containing debate session JSON files (default: ./debate_results)
- `--output-dir, -o`: Output directory for visualizations (default: same as results-dir)
- `--specific-session, -s`: Generate visualizations for a specific session file only
- `--no-performance-matrix`: Skip performance matrix generation
- `--no-consensus-tracking`: Skip consensus tracking visualization
- `--no-agent-comparison`: Skip agent comparison visualization
- `--no-detailed-tracking`: Skip detailed tracking report

## Updated Performance Matrix

The performance matrix now shows accuracy across **phases** instead of rounds:

- **Initial**: Agent's first prediction
- **After Debate**: Agent's accuracy after debating with other agents
- **After Self-Adjustment**: Agent's final accuracy after self-adjustment

This provides a clearer view of how each agent's performance evolves through the debate process.

## Integration with Main System

When running debates without visualization:

```bash
python run_debate.py default game_size=5 game_num=1
```

The system will suggest running the visualization script:

```
💡 To generate visualizations later, run:
   python debate/debate_visualizer.py --results-dir ./debate_results
```

## Generated Files

The visualization script generates:

- `performance_matrix_YYYYMMDD_HHMMSS.png` - Agent accuracy by phase
- `consensus_tracking_YYYYMMDD_HHMMSS.png` - Consensus evolution over rounds
- `agent_comparison_YYYYMMDD_HHMMSS.png` - Agent performance comparison
- `debate_flow_X_YYYYMMDD_HHMMSS.png` - Individual debate flow diagrams
- `summary_report_YYYYMMDD_HHMMSS.txt` - Summary statistics
- `detailed_tracking_YYYYMMDD_HHMMSS.txt` - Detailed agent solution tracking
