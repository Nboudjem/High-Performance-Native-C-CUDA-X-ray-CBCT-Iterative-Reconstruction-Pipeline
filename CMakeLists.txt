cmake_minimum_required(VERSION 3.12)
project(WalnutCT CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_BUILD_RPATH "/usr/local/astra/lib")
set(CMAKE_INSTALL_RPATH "/usr/local/astra/lib")

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

include_directories(
    include
    /content/astra-toolbox/include
    ${OpenCV_INCLUDE_DIRS}
)

link_directories(/usr/local/astra/lib)

add_executable(WalnutCT
    src/main.cpp
    src/NesterovOptimizer.cpp
)

target_link_libraries(WalnutCT
    PRIVATE
        astra
        ${OpenCV_LIBS}
        CUDA::cudart
)

target_include_directories(WalnutCT PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    /content/astra-toolbox/include
    /content/astra-toolbox/include/astra
)
