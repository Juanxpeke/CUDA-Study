# OpenGL Test

add_executable(test_opengl test_opengl.cpp)

set_property(TARGET test_opengl PROPERTY CXX_STANDARD 20)

target_link_libraries(test_opengl PRIVATE ${THIRD_PARTY_LIBRARIES})
target_include_directories(test_opengl PRIVATE ${THIRD_PARTY_INCLUDE_DIRECTORIES})