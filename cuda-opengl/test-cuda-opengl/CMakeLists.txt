# CUDA with OpenGL Test

add_executable(test_cuda_opengl test_cuda_opengl.cu)

target_link_libraries(test_cuda_opengl PRIVATE ${THIRD_PARTY_LIBRARIES})
target_include_directories(test_cuda_opengl PRIVATE ${THIRD_PARTY_INCLUDE_DIRECTORIES} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

