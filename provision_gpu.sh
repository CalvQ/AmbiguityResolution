#! bin/bash

# STEP 1: Terminal 1: Running the service

NUM_GPUS=1
HOURS=1
JOB_NAME=llm_server

# Allocate GPU
srun --partition=GPU-shared --gres="gpu:${NUM_GPUS}" --time="${HOURS}:00:00" --job-name=$JOB_NAME --pty bash