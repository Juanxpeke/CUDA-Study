cmake_minimum_required(VERSION 3.15)
project(cuda-opengl C CXX CUDA)

add_definitions(-D_USE_MATH_DEFINES)

set(THIRD_PARTY_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/glad/include"
									"${CMAKE_CURRENT_SOURCE_DIR}/glfw/include")
set(THIRD_PARTY_LIBRARIES glfw glad)

add_subdirectory(glfw)
add_subdirectory(glad)
add_subdirectory(test-opengl)
add_subdirectory(test-cuda-opengl)