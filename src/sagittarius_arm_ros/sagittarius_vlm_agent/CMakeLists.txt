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
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY launch config nodes
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  USE_SOURCE_PERMISSIONS
)
