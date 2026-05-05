cmake_minimum_required(VERSION 2.8.3)
project(sagittarius_vlm_agent)

find_package(catkin REQUIRED COMPONENTS
  actionlib
  cv_bridge
  geometry_msgs
  rospy
  sensor_msgs
  std_msgs
)

catkin_package()

catkin_install_python(
  PROGRAMS
    nodes/vlm_agent_executor.py
    nodes/camera_intrinsics_calibration.py
    nodes/hand_eye_calibration_tool.py
    nodes/test_hand_eye_calibration.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY launch config nodes
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  USE_SOURCE_PERMISSIONS
)
