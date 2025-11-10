#! bin/bash

# STEP 3: Terminal 2: Port Forwarding

PSC_USER={YOUR PSC USERNAME}
JOB_NAME=llm_server
PORT=8080

# Forward User PSC VM to local
ssh -L "${PORT}:${PSC_USER}:${PORT}" bridges2.psc.edu -l $PSC_USER

# Get GPU VM Name and Forward GPU VM to User PSC VM
export NODE=$(squeue --user=$PSC_USER --name $JOB_NAME -o "%N" -h)
ssh -L "${PORT}:127.0.0.1:${PORT}" $NODE