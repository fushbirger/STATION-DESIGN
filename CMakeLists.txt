cmake_minimum_required(VERSION 3.10)
project(Mini3DEngine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)

add_executable(engine main.cpp)
target_link_libraries(engine PRIVATE OpenGL::GL GLEW::GLEW glfw)
target_compile_definitions(engine PRIVATE ASSET_DIR="${CMAKE_SOURCE_DIR}/assets")