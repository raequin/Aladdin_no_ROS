cmake_minimum_required(VERSION 3.1)
project(VisionServer)

add_definitions(-std=c++11)

set(CMAKE_BUILD_TYPE RELEASE)

set(OpenCV_DIR "../lib/opencv/build")

find_package(OpenCV REQUIRED)

# Stuff for 0mq
## load in pkg-config support
find_package(PkgConfig)
## use pkg-config to get hints for 0mq locations
pkg_check_modules(PC_ZeroMQ QUIET zmq)

## use the hint from above to find where 'zmq.hpp' is located
find_path(ZeroMQ_INCLUDE_DIR
        NAMES zmq.hpp
        PATHS ${PC_ZeroMQ_INCLUDE_DIRS}
        )

## use the hint from about to find the location of libzmq
find_library(ZeroMQ_LIBRARY
        NAMES zmq
        PATHS ${PC_ZeroMQ_LIBRARY_DIRS}
        )

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${ZeroMQ_INCLUDE_DIR}
  ../lib/apriltag/
  )
aux_source_directory(../lib/apriltags/common COMMON_SRC)
aux_source_directory(../lib/apriltags APRILTAG_SRCS)

add_executable(vision_server src/VisionServer.cpp src/CreelTracker.cpp)

target_link_libraries(vision_server
  ${OpenCV_LIBS}
  ${ZeroMQ_LIBRARY}
  flycapture
  flycapturegui
  apriltag
  )
