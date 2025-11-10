#! bin/bash

# STEP 2: Setting up Environment and Running Service

CONDA_ENV_NAME=capstone
PORT=8080

# Load conda environment and install dependencies
module load anaconda3
conda create -n $CONDA_ENV_NAME python=3.11 -y
conda activate $CONDA_ENV_NAME
# pip install -r requirements.txt
pip install fastapi uvicorn

# Run service [Edit according to what you need to run your service]
uvicorn main:app --host 0.0.0.0 --port $PORT