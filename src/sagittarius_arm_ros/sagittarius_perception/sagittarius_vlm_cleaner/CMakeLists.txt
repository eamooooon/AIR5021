cmake_minimum_required(VERSION 2.8.3)
project(sagittarius_vlm_cleaner)

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
    nodes/clean_desk_executor.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY launch config nodes
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  USE_SOURCE_PERMISSIONS
)
