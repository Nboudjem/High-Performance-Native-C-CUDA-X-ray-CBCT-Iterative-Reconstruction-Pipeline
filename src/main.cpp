#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include "NesterovOptimizer.hpp"

// Helper to load geometry text file (scan_geom_corrected.geom)
std::vector<float> loadGeometry(const std::string& filepath, int num_procs) {
    std::ifstream file(filepath);
    std::vector<float> vecs;
    std::string line;
    int count = 0;
    while (std::getline(file, line) && count < num_procs) {
        std::stringstream ss(line);
        float val;
        while (ss >> val) {
            vecs.push_back(val);
        }
        count++;
    }
    return vecs;
}

int main(int argc, char** argv) {
    auto start_time = std::chrono::high_resolution_clock::now();

    std::cout << "========================================" << std::endl;
    std::cout << " Starting Native C++ Walnut CT Pipeline" << std::endl;
    std::cout << "========================================" << std::endl;

    // User defined settings (matching Python script)
    int walnut_id = 1;
    int orbit_id = 1;
    int angular_sub_sampling = 10;
    int voxel_per_mm = 5; // 10 in first version that was killed!

    std::string data_path = "/content/drive/MyDrive";
    std::string recon_path = "/content/drive/MyDrive/Walnut1/cppWalnutsOwnReconstructions/";
    std::string data_path_full = data_path + "/Walnut" + std::to_string(walnut_id) + "/Projections/tubeV" + std::to_string(orbit_id);

    int projs_rows = 972;
    int projs_cols = 768;
    int nb_projs_orbit = 1200 / angular_sub_sampling; // 120 projections

    std::cout << "[C++] Loading flat, dark fields and projection TIFF stack..." << std::endl;

    // Load Dark Field
    std::string dark_path = data_path_full + "/di000000.tif";
    cv::Mat dark = cv::imread(dark_path, cv::IMREAD_UNCHANGED);
    cv::flip(dark, dark, 0); // flipud
    cv::transpose(dark, dark);
    dark.convertTo(dark, CV_32F);

    // Load Flat Fields (average two flats)
    std::string flat1_path = data_path_full + "/io000000.tif";
    std::string flat2_path = data_path_full + "/io000001.tif";
    cv::Mat f1 = cv::imread(flat1_path, cv::IMREAD_UNCHANGED);
    cv::Mat f2 = cv::imread(flat2_path, cv::IMREAD_UNCHANGED);
    cv::flip(f1, f1, 0); cv::transpose(f1, f1); f1.convertTo(f1, CV_32F);
    cv::flip(f2, f2, 0); cv::transpose(f2, f2); f2.convertTo(f2, CV_32F);
    cv::Mat flat = (f1 + f2) * 0.5f;

    // Load Geometry
    std::vector<float> vecs = loadGeometry(data_path_full + "/scan_geom_corrected.geom", 1200);

    // Allocate sinogram buffer matching ASTRA convention: (rows, angles, cols)
    std::vector<float> projs_flat(projs_rows * nb_projs_orbit * projs_cols, 0.0f);

    // Load Projections with Beer-Lambert Law linearization
    for (int i = 0; i < nb_projs_orbit; ++i) {
        int proj_idx = 1200 - (i * angular_sub_sampling);
        char filename[512];
        std::snprintf(filename, sizeof(filename), "%s/scan_%06d.tif", data_path_full.c_str(), proj_idx);

        cv::Mat proj = cv::imread(filename, cv::IMREAD_UNCHANGED);
        if (proj.empty()) {
            proj = cv::Mat::zeros(projs_rows, projs_cols, CV_32F);
        } else {
            cv::flip(proj, proj, 0);
            cv::transpose(proj, proj);
            proj.convertTo(proj, CV_32F);

            // Linearization: (proj - dark) / (flat - dark) -> -log()
            for (int r = 0; r < proj.rows; ++r) {
                for (int c = 0; c < proj.cols; ++c) {
                    float d = dark.at<float>(r, c);
                    float f = flat.at<float>(r, c);
                    float val = proj.at<float>(r, c);
                    val = (val - d) / (f - d);
                    val = -std::log(std::max(val, 1e-5f));
                    proj.at<float>(r, c) = val;
                }
            }
        }

        // Copy into 3D tensor buffer [row, angle, col]
        for (int r = 0; r < projs_rows; ++r) {
            for (int c = 0; c < projs_cols; ++c) {
                int dest_idx = r * (nb_projs_orbit * projs_cols) + i * projs_cols + c;
                projs_flat[dest_idx] = proj.at<float>(r, c);
            }
        }
    }

    int voxel_per_mm_val = voxel_per_mm;
    int vol_sz = 3 * (50 * voxel_per_mm_val + 1); // cubic volume dimensions
    float vox_sz = 1.0f / voxel_per_mm_val;

    // Instantiate and run Optimizer
    NesterovOptimizer optimizer(vol_sz, vox_sz, 0.0f);
    optimizer.initialize(vecs, projs_flat, nb_projs_orbit, projs_rows, projs_cols);

    std::vector<float> reconstructed_volume;
    optimizer.run(50, reconstructed_volume);

    // Save Reconstructed Slices as 16-bit TIFFs
    std::cout << "[C++] Saving reconstructed slices to disk: " << recon_path << std::endl;

    std::string mkdir_cmd = "mkdir -p " + recon_path;
    int ret = system(mkdir_cmd.c_str());

    int slice_count = vol_sz; 
    size_t slice_elements = (size_t)vol_sz * vol_sz;

    // Optional: find global min/max across the volume for proper contrast scaling, 
    // or scale slice-by-slice. Slice-by-slice min-max normalization is safest for viewing:
    for (int i = 0; i < slice_count; ++i) {
        cv::Mat slice(vol_sz, vol_sz, CV_32F, &reconstructed_volume[i * slice_elements]);

        // Find min and max for this slice
        double minVal, maxVal;
        cv::minMaxLoc(slice, &minVal, &maxVal);

        // Avoid division by zero if slice is completely flat/empty
        cv::Mat slice_scaled;
        if (maxVal - minVal > 1e-5f) {
            // Scale to [0, 65535] for 16-bit unsigned integer depth
            slice.convertTo(slice_scaled, CV_16U, 65535.0 / (maxVal - minVal), -minVal * 65535.0 / (maxVal - minVal));
        } else {
            slice.convertTo(slice_scaled, CV_16U);
        }

        char slice_filename[512];
        std::snprintf(slice_filename, sizeof(slice_filename), "%s/nnls_pos1_iter50_ass10_vmm5_%06d.tiff", recon_path.c_str(), i);

        cv::imwrite(slice_filename, slice_scaled);
    }
    std::cout << "[C++] All 16-bit slices saved successfully!" << std::endl;
    
    return 0;
}
