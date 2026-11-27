#ifndef NESTEROV_OPTIMIZER_HPP
#define NESTEROV_OPTIMIZER_HPP

#include <vector>

class NesterovOptimizer {
public:
    NesterovOptimizer(int vol_size, float vox_sz, float min_constraint = 0.0f);
    ~NesterovOptimizer();

    void initialize(const std::vector<float>& geometry_vecs,
                    const std::vector<float>& sinogram_data,
                    int num_projections, int rows, int cols);

    void run(int iterations, std::vector<float>& h_output_volume);

private:
    int vol_size_;
    float vox_sz_;
    float min_constraint_;
    float lipchitz_;
    float nu_;

    int num_projections_;
    int rows_;
    int cols_;

    std::vector<float> geometry_vecs_;
    std::vector<float> sinogram_data_;

    float* d_volume_;
    float* d_sino_;
    float* d_at_y_;
};

#endif // NESTEROV_OPTIMIZER_HPP
