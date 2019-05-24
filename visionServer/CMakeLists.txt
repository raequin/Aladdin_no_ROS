cmake_minimum_required(VERSION 3.1)
project(VisionServer)

add_definitions(-std=c++11)

set(CMAKE_BUILD_TYPE RELEASE)

find_package(OpenCV REQUIRED)
find_package(ZeroMQ 4 REQUIRED PATHS "../lib/libzmq/build")

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${ZeroMQ_INCLUDE_DIRS}
  ../lib/apriltag/
  )
aux_source_directory(../lib/apriltags/common COMMON_SRC)
aux_source_directory(../lib/apriltags APRILTAG_SRCS)

add_executable(vision_server src/VisionServer.cpp src/CreelTracker.cpp)

target_link_libraries(vision_server
  ${OpenCV_LIBRARIES}
  ${ZeroMQ_LIBRARY}
  flycapture
  flycapturegui
  apriltag
  )
