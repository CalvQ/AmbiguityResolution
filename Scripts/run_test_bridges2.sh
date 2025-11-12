#!/bin/bash
#SBATCH --job-name=arengine_test
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100-16:1
#SBATCH --partition=GPU-shared
#SBATCH --time=1:00:00

# ============================================================================
# AREngine Quick Test Job for PSC Bridges2
# ============================================================================
#
# This is a SHORT test job to verify everything works before the full run.
# Tests on just 10 samples to catch any issues quickly.
#
# Usage:
#   sbatch run_test_bridges2.sh
#
# ============================================================================

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET_PATH="data/your_dataset.json"
OUTPUT_CSV="results/test_results.csv"
LOG_FILE="logs/test_${SLURM_JOB_ID}.log"

# Test parameters - small for quick validation
LIMIT=10
CHECKPOINT_EVERY=5

# Python environment
CONDA_ENV="vln"

# ============================================================================
# SCRIPT START
# ============================================================================

echo "=========================================="
echo "AREngine Quick Test Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo ""

# Create directories
mkdir -p logs results

# Load environment
source ~/.bashrc
conda activate $CONDA_ENV

# Check GPU
echo "GPU Check:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Check dataset
if [ ! -f "$DATASET_PATH" ]; then
    echo "ERROR: Dataset not found at $DATASET_PATH"
    exit 1
fi

# Run test
echo "Running test evaluation on $LIMIT samples..."
echo ""

python eval_dataset_checkpoint.py \
    $DATASET_PATH \
    $OUTPUT_CSV \
    --limit $LIMIT \
    --checkpoint-every $CHECKPOINT_EVERY \
    --log-file $LOG_FILE

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Test finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Test passed! Ready for full evaluation."
    echo ""
    echo "Next steps:"
    echo "  1. Check results: cat $OUTPUT_CSV"
    echo "  2. Review log: cat $LOG_FILE"
    echo "  3. If good, run full job: sbatch run_eval_bridges2.sh"
else
    echo "✗ Test failed! Fix issues before running full job."
    echo "  Check error log: cat logs/test_${SLURM_JOB_ID}.err"
fi

exit $EXIT_CODE