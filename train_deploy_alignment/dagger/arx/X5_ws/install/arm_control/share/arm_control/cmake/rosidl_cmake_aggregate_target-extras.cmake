# generated from rosidl_cmake/cmake/rosidl_cmake_aggregate_target-extras.cmake.in

# Create a convenience aggregate target arm_control::arm_control
# that links all generated interface targets, so downstream packages can use
# a single modern CMake target name instead of ${arm_control_TARGETS}.
if(arm_control_TARGETS AND NOT TARGET arm_control::arm_control)
  add_library(arm_control::arm_control INTERFACE IMPORTED)
  set_target_properties(arm_control::arm_control PROPERTIES
    INTERFACE_LINK_LIBRARIES "${arm_control_TARGETS}")
endif()
