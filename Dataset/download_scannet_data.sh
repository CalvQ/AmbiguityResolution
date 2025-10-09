#!/bin/bash

DIRECTORY="scannet_data"

if [ -d "$DIRECTORY" ]; then
  echo "Directory '$DIRECTORY' exists."
else
  mkdir scannet_data
fi

python download-scannetv2.py -o scannet_data --type _vh_clean_2.ply --skip_existing
python download-scannetv2.py -o scannet --type .aggregation.json --skip_existing
python download-scannetv2.py -o scannet --type _vh_clean_2.0.010000.segs.json --skip_existing
