#pragma once

#include <optional>
#include <vector>
#include <cstddef>

#include <astra/VolumeGeometry3D.h>
#include <astra/ConeVecProjectionGeometry3D.h>
#include <astra/Data3D.h>

namespace ct {

struct NesterovOptions {
    int num_iterations = 50;
    int power_iterations = 10;
    std::optional<float> min_constraint = 0.0f;
    std::optional<float> max_constraint = std::nullopt;
    unsigned int seed = 42;
};

class NesterovAcceleratedGradient {
public:
    NesterovAcceleratedGradient(
        const astra::CVolumeGeometry3D& vol_geom,
        const astra::CConeVecProjectionGeometry3D& proj_geom,
        astra::CFloat32ProjectionData3D& sino_data,
        const NesterovOptions& opts = {}
    );

    void run(astra::CFloat32VolumeData3D& vol_data);

    const std::vector<float>& objective_history() const {
        return obj_func_;
    }

private:
    const astra::CVolumeGeometry3D& vol_geom_;
    const astra::CConeVecProjectionGeometry3D& proj_geom_;
    astra::CFloat32ProjectionData3D& sino_data_;

    NesterovOptions opts_;

    float lipschitz_ = 1.0f;
    float nu_ = 1.0f;

    std::vector<float> aty_;
    std::vector<float> obj_func_;

    void forward_project(
        const std::vector<float>& x,
        std::vector<float>& y
    ) const;

    void back_project(
        const std::vector<float>& y,
        std::vector<float>& x
    ) const;

    float compute_lipschitz(
        std::size_t vol_elems,
        std::size_t sino_elems
    );

    void apply_constraints(std::vector<float>& x) const;
};

} // namespace ct