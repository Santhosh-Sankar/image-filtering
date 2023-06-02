cmake_minimum_required(VERSION 3.25)
project(cv_ass1_cuda CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(ENV{OPENCV_DIR} "G:\\opencv\\opencv\\build")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")


find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(cv_ass1_cuda main.cu filters.cu utils.cu display.cu filters.h utils.h)
set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)

target_link_libraries(cv_ass1_cuda ${OpenCV_LIBS})
set_target_properties(cv_ass1_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
