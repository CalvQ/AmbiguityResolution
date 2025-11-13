cmake_minimum_required(VERSION 3.0.2)
project(ai_module)

find_package(catkin REQUIRED COMPONENTS
  rospy
  roscpp
  std_msgs
  sensor_msgs
  geometry_msgs
)

catkin_package()

# Add your Python dependencies
catkin_install_python(PROGRAMS
  AmbiguityLM/ambig_lm.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)