cmake_minimum_required(VERSION 3.1.0)
project(DepthMarkerCPP LANGUAGES CXX C)

set(REALSENSE "C:/Users/robotik/Desktop/RealSense") 
set(OPENCV "C:/Users/robotik/Desktop/opencv/build")

find_package( OpenCV REQUIRED )

add_executable(DepthMarkerCPP src/Main.cpp)
target_link_libraries(DepthMarkerCPP ${REALSENSE}/bin/64bit/realsense2.lib ${OpenCV_LIBS})
include_directories(DepthMarkerCPP ${REALSENSE})
