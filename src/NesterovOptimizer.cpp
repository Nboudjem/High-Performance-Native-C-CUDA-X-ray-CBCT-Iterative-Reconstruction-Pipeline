
#include "NesterovOptimizer.hpp"

#include <astra/Data3D.h>
#include <astra/CudaProjector3D.h>
#include <astra/CudaForwardProjectionAlgorithm3D.h>
#include <astra/CudaBackProjectionAlgorithm3D.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

namespace ct {

namespace {

float l2_norm(const std::vector<float>& v)
{
    double sum_sq = 0.0;

    for (float val : v) {
        sum_sq += static_cast<double>(val) * val;
    }

    return static_cast<float>(std::sqrt(sum_sq));
}


float half_squared_residual_norm(
    const std::vector<float>& wx,
    const float* s,
    std::size_t n)
{
    double sum_sq = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        const double diff =
            static_cast<double>(wx[i]) -
            static_cast<double>(s[i]);

        sum_sq += diff * diff;
    }

    return static_cast<float>(0.5 * sum_sq);
}

} // namespace


// -----------------------------------------------------------------------------
// Constructor
// -----------------------------------------------------------------------------

NesterovAcceleratedGradient::NesterovAcceleratedGradient(
    const astra::CVolumeGeometry3D& vol_geom,
    const astra::CConeVecProjectionGeometry3D& proj_geom,
    astra::CFloat32ProjectionData3D& sino_data,
    const NesterovOptions& opts)

    : vol_geom_(vol_geom),
      proj_geom_(proj_geom),
      sino_data_(sino_data),
      opts_(opts)
{
    const std::size_t vol_elems =
        static_cast<std::size_t>(vol_geom_.getGridTotCount());

    const std::size_t sino_elems =
        static_cast<std::size_t>(proj_geom_.getProjectionCount()) *
        static_cast<std::size_t>(proj_geom_.getDetectorRowCount()) *
        static_cast<std::size_t>(proj_geom_.getDetectorColCount());


    // Compute Lipschitz constant.

    lipschitz_ = compute_lipschitz(vol_elems, sino_elems);

    if (lipschitz_ <= 0.0f ||
        !std::isfinite(lipschitz_)) {
        throw std::runtime_error(
            "Invalid Lipschitz constant. Check ASTRA projector and geometry.");
    }

    nu_ = 1.0f / lipschitz_;


    // Compute A^T * s.

    const float* sino_ptr =
        sino_data_.getFloat32Memory();

    std::vector<float> s_vec(
        sino_ptr,
        sino_ptr + sino_elems);

    aty_.assign(vol_elems, 0.0f);

    back_project(s_vec, aty_);


    std::cout << "Optimizer initialized (nu = "
              << nu_ << ").\n";
}


// -----------------------------------------------------------------------------
// Forward projection: y = A*x
// -----------------------------------------------------------------------------

void NesterovAcceleratedGradient::forward_project(
    const std::vector<float>& x,
    std::vector<float>& y) const
{
    const std::size_t vol_elems =
        static_cast<std::size_t>(vol_geom_.getGridTotCount());

    const std::size_t sino_elems = y.size();

    if (x.size() != vol_elems) {
        throw std::runtime_error(
            "Forward projection: incorrect volume size");
    }

    std::fill(y.begin(), y.end(), 0.0f);


    // Allocate ASTRA data objects.

    std::unique_ptr<astra::CFloat32VolumeData3D> x_tmp(
        astra::createCFloat32VolumeData3DMemory(vol_geom_));

    std::unique_ptr<astra::CFloat32ProjectionData3D> y_tmp(
        astra::createCFloat32ProjectionData3DMemory(proj_geom_));

    if (!x_tmp || !y_tmp) {
        throw std::runtime_error(
            "Failed to allocate ASTRA forward projection data");
    }


    // Copy input volume into ASTRA memory.

    std::copy(
        x.begin(),
        x.end(),
        x_tmp->getFloat32Memory());


    // Create CUDA projector.

    astra::CCudaProjector3D projector(
        proj_geom_,
        vol_geom_);


    // Create and initialize forward projection algorithm.

    astra::CCudaForwardProjectionAlgorithm3D fp;

    if (!fp.initialize(
            &projector,
            y_tmp.get(),
            x_tmp.get())) {

        throw std::runtime_error(
            "Failed to initialize ASTRA CUDA forward projector");
    }


    // Execute forward projection.

    if (!fp.run()) {
        throw std::runtime_error(
            "ASTRA CUDA forward projection failed");
    }


    // Copy projection result back to standard vector.

    const float* result =
        y_tmp->getFloat32Memory();

    std::copy(
        result,
        result + sino_elems,
        y.begin());
}


// -----------------------------------------------------------------------------
// Backprojection: x = A^T*y
// -----------------------------------------------------------------------------

void NesterovAcceleratedGradient::back_project(
    const std::vector<float>& y,
    std::vector<float>& x) const
{
    const std::size_t vol_elems =
        static_cast<std::size_t>(vol_geom_.getGridTotCount());

    const std::size_t sino_elems =
        static_cast<std::size_t>(proj_geom_.getProjectionCount()) *
        static_cast<std::size_t>(proj_geom_.getDetectorRowCount()) *
        static_cast<std::size_t>(proj_geom_.getDetectorColCount());


    if (y.size() != sino_elems ||
        x.size() != vol_elems) {

        throw std::runtime_error(
            "Backprojection: incorrect input or output size");
    }


    std::fill(x.begin(), x.end(), 0.0f);


    // Allocate ASTRA data objects.

    std::unique_ptr<astra::CFloat32ProjectionData3D> y_tmp(
        astra::createCFloat32ProjectionData3DMemory(proj_geom_));

    std::unique_ptr<astra::CFloat32VolumeData3D> x_tmp(
        astra::createCFloat32VolumeData3DMemory(vol_geom_));

    if (!y_tmp || !x_tmp) {
        throw std::runtime_error(
            "Failed to allocate ASTRA backprojection data");
    }


    // Copy input projections into ASTRA memory.

    std::copy(
        y.begin(),
        y.end(),
        y_tmp->getFloat32Memory());


    // Create CUDA projector.

    astra::CCudaProjector3D projector(
        proj_geom_,
        vol_geom_);


    // Create and initialize backprojection algorithm.

    astra::CCudaBackProjectionAlgorithm3D bp;

    if (!bp.initialize(
            &projector,
            y_tmp.get(),
            x_tmp.get())) {

        throw std::runtime_error(
            "Failed to initialize ASTRA CUDA backprojector");
    }


    // Execute backprojection.

    if (!bp.run()) {
        throw std::runtime_error(
            "ASTRA CUDA backprojection failed");
    }


    // Copy reconstructed volume back to vector.

    const float* result =
        x_tmp->getFloat32Memory();

    std::copy(
        result,
        result + vol_elems,
        x.begin());
}


// -----------------------------------------------------------------------------
// Estimate Lipschitz constant using power iteration.
// -----------------------------------------------------------------------------

float NesterovAcceleratedGradient::compute_lipschitz(
    std::size_t vol_elems,
    std::size_t sino_elems)
{
    std::mt19937 rng(opts_.seed);

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);


    std::vector<float> b_k(vol_elems);

    for (float& v : b_k) {
        v = dist(rng);
    }


    std::vector<float> wb(sino_elems, 0.0f);

    std::vector<float> b_k1(vol_elems, 0.0f);

    float norm = 0.0f;


    std::cout
        << "Running power iteration to determine step size\n";


    for (int i = 0; i < opts_.power_iterations; ++i) {

        forward_project(b_k, wb);

        back_project(wb, b_k1);


        norm = l2_norm(b_k1);

        if (norm <= 0.0f ||
            !std::isfinite(norm)) {

            throw std::runtime_error(
                "Power iteration failed: zero or invalid norm");
        }


        for (std::size_t j = 0; j < vol_elems; ++j) {
            b_k[j] = b_k1[j] / norm;
        }
    }

    return norm;
}


// -----------------------------------------------------------------------------
// Apply constraints.
// -----------------------------------------------------------------------------

void NesterovAcceleratedGradient::apply_constraints(
    std::vector<float>& x) const
{
    if (!opts_.min_constraint.has_value() &&
        !opts_.max_constraint.has_value()) {
        return;
    }


    const float lo =
        opts_.min_constraint.value_or(
            -std::numeric_limits<float>::infinity());

    const float hi =
        opts_.max_constraint.value_or(
            std::numeric_limits<float>::infinity());


    for (float& val : x) {
        val = std::clamp(val, lo, hi);
    }
}


// -----------------------------------------------------------------------------
// Nesterov accelerated gradient reconstruction.
// -----------------------------------------------------------------------------

void NesterovAcceleratedGradient::run(
    astra::CFloat32VolumeData3D& vol_data)
{
    const std::size_t vol_elems =
        static_cast<std::size_t>(vol_geom_.getGridTotCount());

    const std::size_t sino_elems =
        static_cast<std::size_t>(proj_geom_.getProjectionCount()) *
        static_cast<std::size_t>(proj_geom_.getDetectorRowCount()) *
        static_cast<std::size_t>(proj_geom_.getDetectorColCount());


    float* v_ptr =
        vol_data.getFloat32Memory();

    const float* s_ptr =
        sino_data_.getFloat32Memory();


    std::vector<float> x_apgd(
        v_ptr,
        v_ptr + vol_elems);

    std::vector<float> x_old = x_apgd;

    std::vector<float> nrm_x(vol_elems, 0.0f);

    std::vector<float> nrm_x_old(vol_elems, 0.0f);

    std::vector<float> gradient(vol_elems, 0.0f);


    for (std::size_t j = 0; j < vol_elems; ++j) {
        gradient[j] = nrm_x[j] - aty_[j];
    }


    std::vector<float> wx(sino_elems, 0.0f);

    obj_func_.assign(
        opts_.num_iterations,
        0.0f);


    int t_acc = 1;

    float running_min_obj =
        std::numeric_limits<float>::infinity();


    std::cout
        << "Running "
        << opts_.num_iterations
        << " iterations of Accelerated Gradient.\n";


    for (int i = 0; i < opts_.num_iterations; ++i) {

        if (i % 10 == 0) {
            std::cout
                << "Iteration "
                << i
                << " / "
                << opts_.num_iterations
                << "\n";
        }


        const float tau =
            static_cast<float>(t_acc - 1) /
            static_cast<float>(t_acc + 2);

        t_acc += 1;


        // Gradient descent step.

        for (std::size_t j = 0; j < vol_elems; ++j) {

            const float descent_dir =
                gradient[j]
                - (tau / nu_) * (x_apgd[j] - x_old[j])
                + tau * (nrm_x[j] - nrm_x_old[j]);


            x_old[j] = x_apgd[j];

            x_apgd[j] -= nu_ * descent_dir;
        }


        // Apply non-negativity or other constraints.

        apply_constraints(x_apgd);


        // Calculate A*x.

        forward_project(x_apgd, wx);


        // Calculate A^T*A*x.

        nrm_x_old = nrm_x;

        back_project(wx, nrm_x);


        // Gradient = A^T*A*x - A^T*s.

        for (std::size_t j = 0; j < vol_elems; ++j) {
            gradient[j] = nrm_x[j] - aty_[j];
        }


        // Objective function:
        // 0.5 * ||A*x - s||^2

        const float obj_val =
            half_squared_residual_norm(
                wx,
                s_ptr,
                sino_elems);


        obj_func_[i] = obj_val;

        running_min_obj =
            std::min(running_min_obj, obj_val);


        // Restart acceleration if objective increases.

        if (obj_val > running_min_obj) {

            t_acc = 1;

            std::cout
                << "Acceleration restarted!\n";
        }
    }


    // Copy result back to ASTRA volume memory.

    std::copy(
        x_apgd.begin(),
        x_apgd.end(),
        v_ptr);
}

} // namespace ct
