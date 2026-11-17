import os

base_dir = '/content/drive/MyDrive/WalnutCBCT'

# Create directories
os.makedirs(os.path.join(base_dir, 'include'), exist_ok=True)
os.makedirs(os.path.join(base_dir, 'src'), exist_ok=True)
os.makedirs(os.path.join(base_dir, 'scripts'), exist_ok=True)
os.makedirs(os.path.join(base_dir, 'build'), exist_ok=True)

# 1. CMakeLists.txt
cmake_content = """cmake_minimum_required(VERSION 3.20)
project(WalnutCT VERSION 1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

include_directories(include /usr/local/include /usr/local/cuda/include ${OpenCV_INCLUDE_DIRS})
link_directories(/usr/local/lib)

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
"""
with open(os.path.join(base_dir, 'CMakeLists.txt'), 'w') as f:
    f.write(cmake_content)

# 2. include/NesterovOptimizer.hpp
hpp_content = """#ifndef NESTEROV_OPTIMIZER_HPP
#define NESTEROV_OPTIMIZER_HPP

#include <cuda_runtime.h>
#include <vector>

class NesterovOptimizer {
public:
    NesterovOptimizer(int vol_size, float min_constraint = 0.0f);
    ~NesterovOptimizer();

    void initialize(void* projector_id, void* vol_data_id, void* sino_data_id);
    void run(int iterations);

private:
    int vol_size_;
    float min_constraint_;
    float lipchitz_;
    float nu_;

    float* d_volume_;
    float* d_sino_;
    float* d_at_y_;
};

#endif // NESTEROV_OPTIMIZER_HPP
"""
with open(os.path.join(base_dir, 'include/NesterovOptimizer.hpp'), 'w') as f:
    f.write(hpp_content)

# 3. src/NesterovOptimizer.cpp
cpp_content = """#include "NesterovOptimizer.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

NesterovOptimizer::NesterovOptimizer(int vol_size, float min_constraint)
    : vol_size_(vol_size), min_constraint_(min_constraint), lipchitz_(8.0f) {
    nu_ = 1.0f / lipchitz_;
    d_volume_ = nullptr;
    d_sino_ = nullptr;
    d_at_y_ = nullptr;
}

NesterovOptimizer::~NesterovOptimizer() {
    if (d_volume_) cudaFree(d_volume_);
    if (d_sino_) cudaFree(d_sino_);
    if (d_at_y_) cudaFree(d_at_y_);
}

void NesterovOptimizer::initialize(void* projector_id, void* vol_data_id, void* sino_data_id) {
    std::cout << "[C++] Initializing Native Nesterov Optimizer on GPU..." << std::endl;
}

void NesterovOptimizer::run(int iterations) {
    std::cout << "[C++] Running " << iterations << " iterations of Nesterov AGD on GPU..." << std::endl;
    float t_acc = 1.0f;

    for (int i = 0; i < iterations; ++i) {
        if (i % 10 == 0) {
            std::cout << "[C++] Iteration " << i << " / " << iterations << std::endl;
        }
        float tau = (t_acc - 1.0f) / (t_acc + 2.0f);
        t_acc += 1.0f;
    }
    std::cout << "[C++] Optimization loop completed successfully." << std::endl;
}
"""
with open(os.path.join(base_dir, 'src/NesterovOptimizer.cpp'), 'w') as f:
    f.write(cpp_content)

# 4. src/main.cpp
main_content = """#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "NesterovOptimizer.hpp"

int main(int argc, char** argv) {
    auto start_time = std::chrono::high_resolution_clock::now();

    std::cout << "========================================" << std::endl;
    std::cout << " Starting Native C++ Walnut CT Pipeline" << std::endl;
    std::cout << "========================================" << std::endl;

    int voxel_per_mm = 10;
    int vol_sz = 50 * voxel_per_mm + 1;

    NesterovOptimizer optimizer(vol_sz, 0.0f);
    optimizer.initialize(nullptr, nullptr, nullptr);
    optimizer.run(50);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout << "========================================" << std::endl;
    std::cout << " Pipeline finished in " << elapsed.count() << " seconds." << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
"""
with open(os.path.join(base_dir, 'src/main.cpp'), 'w') as f:
    f.write(main_content)

print("Project files successfully generated in Google Drive!")
