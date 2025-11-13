#! bin/bash

# Get simulator code
DIRECTORY="CMU-VLA-Challenge"
if [ -d "$DIRECTORY" ]; then
  echo "Directory '$DIRECTORY' exists"
else
  echo "Directory '$DIRECTORY' does not exist, cloning repo..."
  # git clone https://github.com/HaochenZ11/CMU-VLA-Challenge.git
  git clone git@github.com:HaochenZ11/CMU-VLA-Challenge.git
fi

# Get environment files (ScanNet) from VLA-3D
DIRECTORY="Dataset/scannet_data"

if [ -d "$DIRECTORY" ]; then
  DIRECTORY="CMU-VLA-Challenge/system/unity/src/vehicle_simulator/mesh/unity/scene0000_00"
  if [ -d "$DIRECTORY" ]; then
    echo "Data copied"
  else
    echo "Copying data"
    cp -r Dataset/scannet_data/. CMU-VLA-Challenge/system/unity/src/vehicle_simulator/mesh/unity
  fi
else
  echo "Directory '$DIRECTORY' does not exist, downloading data..."
  bash Dataset/download_scannet_data.sh
  echo "Copying data"
  cp -r Dataset/scannet_data/. CMU-VLA-Challenge/system/unity/src/vehicle_simulator/mesh/unity
fi

# Replace default "dummy_vlm" with our ambiguity language model
rm -r CMU-VLA-Challenge/ai_module/src/*
cp -r DemoFiles/. CMU-VLA-Challenge/ai_module/src

# Run demo code
cd CMU-VLA-Challenge
docker compose -f docker/compose.yml up --build

