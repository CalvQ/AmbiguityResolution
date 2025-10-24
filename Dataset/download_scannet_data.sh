#!/bin/bash

DIRECTORY="scannet_data"

# Get IRef-VLA Scannet Data
python download_dataset.py
unzip Scannet.zip
rm Scannet.zip
mv "Scannet" $DIRECTORY

# Get ScanRefer Data
gdown https://drive.google.com/uc?id=1x9PcZctaLLC79vF42ktl-bNRnixrKO15
unzip scanrefer.zip
rm scanrefer.zip
rm -r __MACOSX