cmake_minimum_required(VERSION 3.20)
project(WalnutCBCT VERSION 1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# Workaround for GCC 13+ variant compatibility with older ASTRA headers
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13)
    add_compile_options(-fpermissive)
endif()

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Include directories including ASTRA's prefix path
include_directories(
    include 
    /usr/local/include 
    /usr/local/cuda/include 
    /usr/local/astra/include
    ${OpenCV_INCLUDE_DIRS}
)

# Link directories pointing to ASTRA's installed library folder
link_directories(
    /usr/local/lib
    /usr/local/astra/lib
)

add_executable(WalnutCT 
    src/NesterovOptimizer.cpp 
    src/main.cpp
)

target_link_libraries(WalnutCT 
    PRIVATE 
    astra 
    ${OpenCV_LIBS} 
    CUDA::cudart 
    CUDA::cublas
)
