#!/bin/bash
# ============================================================================
# Setup Helper for PSC Batch Jobs
# ============================================================================
# This script helps you configure the batch job scripts for your specific setup.

echo "=========================================="
echo "AREngine Evaluation - Setup Helper"
echo "=========================================="
echo ""

# Get current directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Function to prompt with default
prompt_with_default() {
    local prompt=$1
    local default=$2
    local varname=$3
    
    echo -n "$prompt [$default]: "
    read value
    if [ -z "$value" ]; then
        eval "$varname='$default'"
    else
        eval "$varname='$value'"
    fi
}

# Function to validate file exists
validate_file() {
    local file=$1
    if [ ! -f "$file" ]; then
        echo "  ⚠️  Warning: File not found: $file"
        echo "      Make sure to upload it before submitting the job"
        return 1
    else
        echo "  ✓ File found: $file"
        return 0
    fi
}

echo "This helper will configure your batch job scripts."
echo ""

# ============================================================================
# Collect configuration
# ============================================================================

echo "Dataset Configuration:"
echo "----------------------"
prompt_with_default "Dataset JSON path" "data/dataset.json" DATASET_PATH
validate_file "$DATASET_PATH"
echo ""

prompt_with_default "Output CSV path" "results/eval_results.csv" OUTPUT_CSV
echo "  Output will be saved to: $OUTPUT_CSV"
echo ""

echo "Environment Configuration:"
echo "--------------------------"
prompt_with_default "Conda environment name" "vln" CONDA_ENV
echo "  Will use: conda activate $CONDA_ENV"
echo ""

prompt_with_default "Your email for notifications" "youremail@andrew.cmu.edu" USER_EMAIL
echo ""

echo "Job Parameters:"
echo "---------------"
prompt_with_default "Checkpoint every N samples" "50" CHECKPOINT_EVERY
echo "  Will save checkpoint every $CHECKPOINT_EVERY samples"
echo ""

echo "GPU Configuration:"
echo "------------------"
echo "Available GPU types:"
echo "  1) v100-16:1 (16GB, faster queue)"
echo "  2) v100-32:1 (32GB, recommended)"
prompt_with_default "Choose GPU type [1-2]" "2" GPU_CHOICE

case $GPU_CHOICE in
    1)
        GPU_TYPE="v100-16:1"
        ;;
    2|*)
        GPU_TYPE="v100-32:1"
        ;;
esac
echo "  Selected: $GPU_TYPE"
echo ""

prompt_with_default "Time limit (hours)" "12" TIME_HOURS
echo "  Job will run for up to $TIME_HOURS hours"
echo ""

# ============================================================================
# Update scripts
# ============================================================================

echo ""
echo "Updating scripts..."
echo ""

# Update main evaluation script
if [ -f "run_eval_bridges2.sh" ]; then
    cp run_eval_bridges2.sh run_eval_bridges2.sh.backup
    
    sed -i "s|^DATASET_PATH=.*|DATASET_PATH=\"$DATASET_PATH\"|" run_eval_bridges2.sh
    sed -i "s|^OUTPUT_CSV=.*|OUTPUT_CSV=\"$OUTPUT_CSV\"|" run_eval_bridges2.sh
    sed -i "s|^CONDA_ENV=.*|CONDA_ENV=\"$CONDA_ENV\"|" run_eval_bridges2.sh
    sed -i "s|^CHECKPOINT_EVERY=.*|CHECKPOINT_EVERY=$CHECKPOINT_EVERY|" run_eval_bridges2.sh
    sed -i "s|#SBATCH --mail-user=.*|#SBATCH --mail-user=$USER_EMAIL|" run_eval_bridges2.sh
    sed -i "s|#SBATCH --gres=gpu:.*|#SBATCH --gres=gpu:$GPU_TYPE|" run_eval_bridges2.sh
    sed -i "s|#SBATCH --time=.*|#SBATCH --time=${TIME_HOURS}:00:00|" run_eval_bridges2.sh
    
    echo "✓ Updated run_eval_bridges2.sh"
    echo "  Backup saved as: run_eval_bridges2.sh.backup"
else
    echo "✗ run_eval_bridges2.sh not found!"
fi

# Update test script
if [ -f "run_test_bridges2.sh" ]; then
    cp run_test_bridges2.sh run_test_bridges2.sh.backup
    
    sed -i "s|^DATASET_PATH=.*|DATASET_PATH=\"$DATASET_PATH\"|" run_test_bridges2.sh
    sed -i "s|^CONDA_ENV=.*|CONDA_ENV=\"$CONDA_ENV\"|" run_test_bridges2.sh
    
    echo "✓ Updated run_test_bridges2.sh"
    echo "  Backup saved as: run_test_bridges2.sh.backup"
else
    echo "✗ run_test_bridges2.sh not found!"
fi

echo ""

# ============================================================================
# Create directories
# ============================================================================

echo "Creating directories..."
mkdir -p logs
mkdir -p results
mkdir -p "$(dirname $DATASET_PATH)"
mkdir -p "$(dirname $OUTPUT_CSV)"
echo "✓ Created: logs, results, data directories"
echo ""

# ============================================================================
# Verify setup
# ============================================================================

echo "=========================================="
echo "Setup Verification"
echo "=========================================="
echo ""

# Check files
echo "Files:"
validate_file "run_eval_bridges2.sh"
validate_file "run_test_bridges2.sh"
validate_file "eval_dataset_checkpoint.py"
validate_file "arengine.py"
validate_file "$DATASET_PATH"
echo ""

# Check conda environment
echo "Conda environment:"
if conda env list | grep -q "^$CONDA_ENV "; then
    echo "  ✓ Conda environment '$CONDA_ENV' exists"
else
    echo "  ⚠️  Conda environment '$CONDA_ENV' not found"
    echo "      Create it with: conda create -n $CONDA_ENV python=3.10"
fi
echo ""

# Make scripts executable
chmod +x run_eval_bridges2.sh run_test_bridges2.sh 2>/dev/null
echo "✓ Made scripts executable"
echo ""

# ============================================================================
# Summary
# ============================================================================

echo "=========================================="
echo "Configuration Summary"
echo "=========================================="
echo ""
echo "Dataset:         $DATASET_PATH"
echo "Output:          $OUTPUT_CSV"
echo "Conda env:       $CONDA_ENV"
echo "Email:           $USER_EMAIL"
echo "Checkpoint:      Every $CHECKPOINT_EVERY samples"
echo "GPU:             $GPU_TYPE"
echo "Time limit:      $TIME_HOURS hours"
echo ""

# ============================================================================
# Next steps
# ============================================================================

echo "=========================================="
echo "Next Steps"
echo "=========================================="
echo ""
echo "1. Run quick test (5-10 minutes):"
echo "   sbatch run_test_bridges2.sh"
echo ""
echo "2. Check test results:"
echo "   cat results/test_results.csv"
echo ""
echo "3. If test passes, run full evaluation:"
echo "   sbatch run_eval_bridges2.sh"
echo ""
echo "4. Monitor progress:"
echo "   squeue -u \$USER"
echo "   tail -f logs/eval_*.out"
echo ""
echo "5. If interrupted, resume:"
echo "   Edit run_eval_bridges2.sh: RESUME=\"--resume\""
echo "   sbatch run_eval_bridges2.sh"
echo ""
echo "=========================================="
echo ""

# Save configuration
cat > setup_config.txt <<EOF
# AREngine Evaluation Configuration
# Generated: $(date)

DATASET_PATH=$DATASET_PATH
OUTPUT_CSV=$OUTPUT_CSV
CONDA_ENV=$CONDA_ENV
USER_EMAIL=$USER_EMAIL
CHECKPOINT_EVERY=$CHECKPOINT_EVERY
GPU_TYPE=$GPU_TYPE
TIME_HOURS=$TIME_HOURS
EOF

echo "Configuration saved to: setup_config.txt"
echo ""
echo "Setup complete! You're ready to submit jobs."