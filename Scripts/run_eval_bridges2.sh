#!/bin/bash
#SBATCH --job-name=arengine_eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100-32:1
#SBATCH --partition=GPU-shared
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email@example.com

# ============================================================================
# AREngine Evaluation Batch Job for PSC Bridges2
# ============================================================================
#
# This script runs the AREngine evaluation with automatic checkpointing.
# If the job fails or times out, you can resubmit with --resume flag.
#
# Usage:
#   1. Edit the parameters below
#   2. sbatch run_eval_bridges2.sh
#   3. If interrupted: edit to add --resume, then resubmit
#
# Monitor:
#   squeue -u $USER
#   tail -f logs/eval_<jobid>.out
#
# ============================================================================

# ============================================================================
# CONFIGURATION - EDIT THESE
# ============================================================================

# Dataset and output paths
DATASET_PATH="data/your_dataset.json"
OUTPUT_CSV="results/eval_results.csv"
LOG_FILE="logs/eval_${SLURM_JOB_ID}.log"

# Evaluation parameters
LIMIT=""  # Leave empty for full dataset, or set to number like "100"
CHECKPOINT_EVERY=50  # Save checkpoint every N samples
RESUME=""  # Set to "--resume" to resume from checkpoint

# Python environment
CONDA_ENV="vln"  # Your conda environment name
# Or use: MODULE_PYTHON="python/3.10.2"  # If using modules instead

# ============================================================================
# SCRIPT START - NO NEED TO EDIT BELOW
# ============================================================================

echo "=========================================="
echo "AREngine Evaluation Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "=========================================="
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs
mkdir -p results

# Print configuration
echo "Configuration:"
echo "  Dataset: $DATASET_PATH"
echo "  Output: $OUTPUT_CSV"
echo "  Log: $LOG_FILE"
echo "  Checkpoint every: $CHECKPOINT_EVERY samples"
echo "  Resume: ${RESUME:-false}"
if [ -n "$LIMIT" ]; then
    echo "  Limit: $LIMIT samples"
else
    echo "  Limit: Full dataset"
fi
echo ""

# Load environment
echo "Loading environment..."
if [ -n "$CONDA_ENV" ]; then
    # Using conda
    source ~/.bashrc
    conda activate $CONDA_ENV
    echo "  Conda environment: $CONDA_ENV"
elif [ -n "$MODULE_PYTHON" ]; then
    # Using modules
    module load $MODULE_PYTHON
    module load cuda/11.7.1  # Adjust CUDA version as needed
    echo "  Python module: $MODULE_PYTHON"
fi

# Verify GPU
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# Print Python environment info
echo "Python environment:"
which python
python --version
echo ""

# Check if dataset exists
if [ ! -f "$DATASET_PATH" ]; then
    echo "ERROR: Dataset not found at $DATASET_PATH"
    exit 1
fi

# Build command
CMD="python eval_dataset_checkpoint.py $DATASET_PATH $OUTPUT_CSV"
CMD="$CMD --checkpoint-every $CHECKPOINT_EVERY"
CMD="$CMD --log-file $LOG_FILE"

if [ -n "$LIMIT" ]; then
    CMD="$CMD --limit $LIMIT"
fi

if [ -n "$RESUME" ]; then
    CMD="$CMD --resume"
fi

echo "Running command:"
echo "  $CMD"
echo ""
echo "=========================================="
echo "Starting evaluation..."
echo "=========================================="
echo ""

# Run evaluation with error handling
$CMD
EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job finished"
echo "=========================================="
echo "Exit code: $EXIT_CODE"
echo "Ended: $(date)"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Evaluation completed successfully!"
    echo "  Results: $OUTPUT_CSV"
    echo "  Log: $LOG_FILE"
else
    echo "✗ Evaluation failed with exit code $EXIT_CODE"
    echo "  Check logs: logs/eval_${SLURM_JOB_ID}.err"
    echo ""
    echo "To resume from checkpoint, edit this script to set:"
    echo "  RESUME=\"--resume\""
    echo "Then resubmit: sbatch $(basename $0)"
fi

exit $EXIT_CODE