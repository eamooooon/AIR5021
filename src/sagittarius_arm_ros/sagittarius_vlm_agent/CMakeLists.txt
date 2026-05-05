cmake_minimum_required(VERSION 2.8.3)
project(sagittarius_vlm_agent)

find_package(catkin REQUIRED COMPONENTS
  actionlib
  actionlib_msgs
  cv_bridge
  geometry_msgs
  message_generation
  rospy
  sensor_msgs
  std_msgs
)

add_action_files(
  FILES
  SGRCtrl.action
)

generate_messages(
  DEPENDENCIES
  actionlib_msgs
  geometry_msgs
  sensor_msgs
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS
    actionlib
    actionlib_msgs
    geometry_msgs
    message_runtime
    rospy
    sensor_msgs
    std_msgs
)

catkin_install_python(
  PROGRAMS
    nodes/vlm_agent_executor.py
    nodes/sgr_ctrl.py
    nodes/camera_intrinsics_calibration.py
    nodes/hand_eye_calibration_tool.py
    nodes/test_hand_eye_calibration.py
    nodes/verify_calibration.py
    nodes/live_verify_calibration.py
    nodes/interactive_hand_eye_verify.py
    nodes/collect_hand_eye_samples.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY launch config nodes
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  USE_SOURCE_PERMISSIONS
)
