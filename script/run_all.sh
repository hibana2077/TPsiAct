#!/bin/bash
# Run all TPsiAct experiments locally (not for PBS)
# This script is for local development and testing

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

echo "============================================================"
echo "          TPsiAct Full Experiment Suite"
echo "============================================================"
echo "Start time: $(date)"
echo ""

# Create experiments directory
mkdir -p experiments

# Run experiments sequentially
EXPERIMENTS=("T000" "T001" "T002" "T003" "T004" "T005")

for exp in "${EXPERIMENTS[@]}"; do
    echo ""
    echo "------------------------------------------------------------"
    echo "Running experiment: $exp"
    echo "------------------------------------------------------------"
    
    if [ -f "script/T/${exp}.sh" ]; then
        # Extract the python command from the shell script and run it
        # For local testing, we might want to reduce epochs
        bash "script/T/${exp}.sh" &
        wait $!
        
        echo "Completed: $exp"
    else
        echo "Warning: script/T/${exp}.sh not found, skipping"
    fi
done

echo ""
echo "============================================================"
echo "          All Experiments Completed"
echo "============================================================"
echo "End time: $(date)"
echo ""
echo "Results are saved in ./experiments/"
echo "To generate final report, run: python src/generate_report.py"
