#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

namespace fs = std::filesystem;

// ================= USER DEFINED SETTINGS =================

// Select walnut sample
int walnut_id = 1;

// Select orbits
std::vector<int> orbits_to_recon = {1, 2, 3};

// Angular subsampling factor
int angular_sub_sampling = 10;

// Voxels per mm
int voxel_per_mm = 10;

// Dataset paths
std::string data_path =
    "/home/nboudjem/Downloads";

std::string recon_path =
    "/home/nboudjem/Downloads/Walnut1/WalnutsOwnReconstructions/";

// Projection dimensions
const int projs_rows = 972;
const int projs_cols = 768;
const int total_projections = 1200;


// ================= HELPER FUNCTIONS =================

// Read image as float
cv::Mat readImageFloat(const std::string& filename)
{
    cv::Mat image = cv::imread(
        filename,
        cv::IMREAD_UNCHANGED
    );

    if (image.empty())
        throw std::runtime_error("Cannot read: " + filename);

    cv::Mat float_image;
    image.convertTo(float_image, CV_32F);

    return float_image;
}


// Apply transformation:
// transpose(flipud(image))
cv::Mat transformImage(const cv::Mat& image)
{
    cv::Mat flipped, transformed;

    cv::flip(image, flipped, 0);
    cv::transpose(flipped, transformed);

    return transformed;
}


// Load geometry file
Eigen::MatrixXf loadGeometry(
    const std::string& filename)
{
    std::vector<float> values;

    std::ifstream file(filename);

    if (!file.is_open())
        throw std::runtime_error(
            "Cannot open geometry file: " + filename
        );

    float value;

    while (file >> value)
        values.push_back(value);

    if (values.size() % 12 != 0)
        throw std::runtime_error(
            "Invalid geometry file format."
        );

    int rows = values.size() / 12;

    Eigen::MatrixXf geometry(rows, 12);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < 12; j++)
        {
            geometry(i, j) = values[i * 12 + j];
        }
    }

    return geometry;
}


// ================= MAIN =================

int main()
{
    try
    {
        auto start = std::chrono::high_resolution_clock::now();

        std::cout << "Load and preprocess data"
                  << std::endl;

        // Dataset directory
        fs::path data_path_full =
            fs::path(data_path) /
            ("Walnut" + std::to_string(walnut_id)) /
            "Projections";

        // File names
        std::string projs_name = "scan_";
        std::string dark_name = "di000000.tif";

        std::vector<std::string> flat_name = {
            "io000000.tif",
            "io000001.tif"
        };

        std::string vecs_name =
            "scan_geom_corrected.geom";

        // Number of projections per orbit
        int nb_projs_orbit =
            (total_projections + angular_sub_sampling - 1)
            / angular_sub_sampling;

        // Projection indices (reverse order)
        std::vector<int> projs_idx;

        for (int i = total_projections;
             i > 0;
             i -= angular_sub_sampling)
        {
            projs_idx.push_back(i);
        }

        // Projection storage:
        // ASTRA convention: rows x total_angles x columns
        std::vector<cv::Mat> projections;

        // Geometry storage
        std::vector<Eigen::VectorXf> vecs;

        // Loop over selected orbits
        for (int orbit_id : orbits_to_recon)
        {
            std::cout << "Loading orbit "
                      << orbit_id << std::endl;

            fs::path orbit_data_path =
                data_path_full /
                ("tubeV" + std::to_string(orbit_id));

            // Load geometry
            Eigen::MatrixXf vecs_orbit =
                loadGeometry(
                    (orbit_data_path / vecs_name).string()
                );

            // Select geometry rows
            for (int i = 0; i < total_projections;
                 i += angular_sub_sampling)
            {
                vecs.push_back(vecs_orbit.row(i));
            }

            // Load dark field
            cv::Mat dark = transformImage(
                readImageFloat(
                    (orbit_data_path / dark_name).string()
                )
            );

            // Load and average flat fields
            cv::Mat flat = cv::Mat::zeros(
                projs_cols,
                projs_rows,
                CV_32F
            );

            for (const auto& filename : flat_name)
            {
                cv::Mat image = transformImage(
                    readImageFloat(
                        (orbit_data_path / filename).string()
                    )
                );

                flat += image / 2.0f;
            }

            // Process projections
            for (int i = 0; i < nb_projs_orbit; i++)
            {
                // Format projection filename
                std::ostringstream filename;

                filename << projs_name
                         << std::setw(6)
                         << std::setfill('0')
                         << projs_idx[i]
                         << ".tif";

                cv::Mat projection = transformImage(
                    readImageFloat(
                        (orbit_data_path / filename.str()).string()
                    )
                );

                // Dark field subtraction
                projection -= dark;

                // Flat field correction
                cv::Mat denominator = flat - dark;

                cv::divide(
                    projection,
                    denominator,
                    projection
                );

                // Negative logarithm
                cv::max(
                    projection,
                    1e-10,
                    projection
                );

                cv::log(projection, projection);

                projection *= -1.0f;

                // Store projection
                projections.push_back(projection);
            }
        }

        // ================= OUTPUT =================

        std::cout << "Total projections loaded: "
                  << projections.size()
                  << std::endl;

        std::cout << "Geometry vectors loaded: "
                  << vecs.size()
                  << std::endl;

        auto end = std::chrono::high_resolution_clock::now();

        double elapsed =
            std::chrono::duration<double>(
                end - start
            ).count();

        std::cout << std::fixed
                  << std::setprecision(3)
                  << elapsed
                  << " sec elapsed"
                  << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: "
                  << e.what()
                  << std::endl;

        return 1;
    }

    return 0;
}
