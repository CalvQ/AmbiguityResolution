#! bin/bash

# Get simulator code
git clone https://github.com/HaochenZ11/CMU-VLA-Challenge.git

# Get environment files (ScanNet) from VLA-3D
DIRECTORY="Dataset/scannet_data"

if [ -d "$DIRECTORY" ]; then
  echo "Directory '$DIRECTORY' exists"
else
  echo "Directory '$DIRECTORY' does not exist, downloading data..."
  bash Dataset/download_scannet_data.sh
fi

echo "Copying data"
cp -r Dataset/scannet_data/. CMU-VLA-Challenge/system/unity/src/vehicle_simulator/mesh/unity

# Replace default "dummy_vlm" with our ambiguity language model
cp -r DemoFiles/AmbiguityLM/. CMU-VLA-Challenge/ai_module/src
rm -r CMU-VLA-Challenge/ai_module/src/dummy_vlm

# Build model package for ROS
cd CMU-VLA-Challenge/ai_module/src
cd ai_module/src/
catkin_create_pkg AmbiguityLM rospy roscpp std_msgs sensor_msgs geometry_msgs

