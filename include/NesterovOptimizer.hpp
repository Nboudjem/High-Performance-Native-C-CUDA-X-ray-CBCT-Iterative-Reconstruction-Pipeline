#ifndef NESTEROV_OPTIMIZER_HPP
#define NESTEROV_OPTIMIZER_HPP

#include <cuda_runtime.h>
#include <vector>
#include <string>

class NesterovOptimizer {
public:
    NesterovOptimizer(int vol_size, float vox_sz, float min_constraint = 0.0f);
    ~NesterovOptimizer();

    // Initialize geometry, projector, and bind data buffers
    void initialize(const std::vector<float>& geometry_vecs, 
                    const std::vector<float>& sinogram_data, 
                    int num_projections, int rows, int cols);

    // Run Nesterov Accelerated Gradient Descent iterations
    void run(int iterations, std::vector<float>& h_output_volume);

private:
    int vol_size_;
    float vox_sz_;
    float min_constraint_;
    float lipchitz_;
    float nu_;

    float* d_volume_;
    float* d_sino_;
    float* d_at_y_;
    
    // Internal helper for power iteration step size estimation
    void estimateLipschitz();
};

#endif // NESTEROV_OPTIMIZER_HPP
