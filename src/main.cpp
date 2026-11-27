
#include "NesterovOptimizer.hpp"

#include <astra/Data3D.h>
#include <astra/GeometryUtil3D.h>
#include <astra/ConeVecProjectionGeometry3D.h>
#include <astra/VolumeGeometry3D.h>

#include <tiffio.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

// -----------------------------------------------------------------------------
// Read TIFF image as float32.
//
// Actual TIFF dimensions:
//     rows = 768
//     cols = 972
//
// No transpose and no vertical flip are applied.
// The returned buffer is row-major:
//     image[row * cols + col]
// -----------------------------------------------------------------------------

std::vector<float> read_tiff(
    const fs::path& file_path,
    int expected_rows,
    int expected_cols)
{
    TIFF* tif = TIFFOpen(file_path.c_str(), "r");

    if (!tif) {
        throw std::runtime_error(
            "Failed to open TIFF file: " + file_path.string());
    }

    uint32_t width = 0;
    uint32_t height = 0;

    uint16_t sample_format = SAMPLEFORMAT_UINT;
    uint16_t bits_per_sample = 16;
    uint16_t samples_per_pixel = 1;
    uint16_t planar_config = PLANARCONFIG_CONTIG;
    uint16_t orientation = ORIENTATION_TOPLEFT;

    if (!TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width) ||
        !TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height)) {

        TIFFClose(tif);

        throw std::runtime_error(
            "Cannot read TIFF dimensions: " + file_path.string());
    }

    TIFFGetFieldDefaulted(
        tif, TIFFTAG_SAMPLEFORMAT, &sample_format);

    TIFFGetFieldDefaulted(
        tif, TIFFTAG_BITSPERSAMPLE, &bits_per_sample);

    TIFFGetFieldDefaulted(
        tif, TIFFTAG_SAMPLESPERPIXEL, &samples_per_pixel);

    TIFFGetFieldDefaulted(
        tif, TIFFTAG_PLANARCONFIG, &planar_config);

    TIFFGetFieldDefaulted(
        tif, TIFFTAG_ORIENTATION, &orientation);


    const int rows = static_cast<int>(height);
    const int cols = static_cast<int>(width);


    if (rows != expected_rows || cols != expected_cols) {

        TIFFClose(tif);

        std::ostringstream msg;

        msg << "Unexpected TIFF dimensions in "
            << file_path.string()
            << ". Expected "
            << expected_rows << " x " << expected_cols
            << ", received "
            << rows << " x " << cols;

        throw std::runtime_error(msg.str());
    }


    if (samples_per_pixel != 1) {

        TIFFClose(tif);

        throw std::runtime_error(
            "Only single-channel TIFF images are supported: " +
            file_path.string());
    }


    if (planar_config != PLANARCONFIG_CONTIG) {

        TIFFClose(tif);

        throw std::runtime_error(
            "Unsupported TIFF planar configuration: " +
            file_path.string());
    }


    if (TIFFIsTiled(tif)) {

        TIFFClose(tif);

        throw std::runtime_error(
            "Tiled TIFF is not supported: " +
            file_path.string());
    }


    // The TIFFs supplied by the user are uint16.
    // Also support uint8 and float32.

    const bool is_uint8 =
        sample_format == SAMPLEFORMAT_UINT &&
        bits_per_sample == 8;

    const bool is_uint16 =
        sample_format == SAMPLEFORMAT_UINT &&
        bits_per_sample == 16;

    const bool is_float32 =
        sample_format == SAMPLEFORMAT_IEEEFP &&
        bits_per_sample == 32;


    if (!is_uint8 && !is_uint16 && !is_float32) {

        TIFFClose(tif);

        throw std::runtime_error(
            "Unsupported TIFF pixel format in: " +
            file_path.string());
    }


    const std::size_t pixel_count =
        static_cast<std::size_t>(rows) * cols;


    std::vector<float> image(pixel_count, 0.0f);


    const tmsize_t scanline_size = TIFFScanlineSize(tif);

    const std::size_t bytes_per_pixel =
        static_cast<std::size_t>(bits_per_sample / 8);

    const std::size_t required_scanline_size =
        static_cast<std::size_t>(cols) * bytes_per_pixel;


    if (scanline_size < 0 ||
        static_cast<std::size_t>(scanline_size) <
            required_scanline_size) {

        TIFFClose(tif);

        throw std::runtime_error(
            "Invalid TIFF scanline size: " +
            file_path.string());
    }


    std::vector<uint8_t> scanline(
        static_cast<std::size_t>(scanline_size));


    for (int r = 0; r < rows; ++r) {

        if (TIFFReadScanline(
                tif,
                scanline.data(),
                static_cast<uint32_t>(r),
                0) < 0) {

            TIFFClose(tif);

            throw std::runtime_error(
                "Failed reading TIFF scanline: " +
                file_path.string());
        }


        for (int c = 0; c < cols; ++c) {

            float value = 0.0f;

            const std::size_t offset =
                static_cast<std::size_t>(c) * bytes_per_pixel;


            if (is_uint8) {

                value = static_cast<float>(
                    scanline[offset]);

            } else if (is_uint16) {

                uint16_t pixel = 0;

                std::memcpy(
                    &pixel,
                    scanline.data() + offset,
                    sizeof(uint16_t));

                value = static_cast<float>(pixel);

            } else if (is_float32) {

                float pixel = 0.0f;

                std::memcpy(
                    &pixel,
                    scanline.data() + offset,
                    sizeof(float));

                value = pixel;
            }


            image[
                static_cast<std::size_t>(r) * cols + c
            ] = value;
        }
    }


    TIFFClose(tif);

    return image;
}


// -----------------------------------------------------------------------------
// Write a 2D float32 TIFF.
// -----------------------------------------------------------------------------

void write_float_tiff_2d(
    const fs::path& file_path,
    const float* data,
    int rows,
    int cols)
{
    TIFF* tif = TIFFOpen(file_path.c_str(), "w");

    if (!tif) {
        throw std::runtime_error(
            "Failed to create output TIFF: " +
            file_path.string());
    }


    TIFFSetField(
        tif, TIFFTAG_IMAGEWIDTH,
        static_cast<uint32_t>(cols));

    TIFFSetField(
        tif, TIFFTAG_IMAGELENGTH,
        static_cast<uint32_t>(rows));

    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);

    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 32);

    TIFFSetField(
        tif, TIFFTAG_SAMPLEFORMAT,
        SAMPLEFORMAT_IEEEFP);

    TIFFSetField(
        tif, TIFFTAG_ORIENTATION,
        ORIENTATION_TOPLEFT);

    TIFFSetField(
        tif, TIFFTAG_PLANARCONFIG,
        PLANARCONFIG_CONTIG);

    TIFFSetField(
        tif, TIFFTAG_COMPRESSION,
        COMPRESSION_NONE);

    TIFFSetField(
        tif, TIFFTAG_PHOTOMETRIC,
        PHOTOMETRIC_MINISBLACK);


    for (int r = 0; r < rows; ++r) {

        const float* row_ptr =
            data + static_cast<std::size_t>(r) * cols;

        if (TIFFWriteScanline(
                tif,
                const_cast<float*>(row_ptr),
                static_cast<uint32_t>(r),
                0) < 0) {

            TIFFClose(tif);

            throw std::runtime_error(
                "Failed writing TIFF scanline: " +
                file_path.string());
        }
    }


    TIFFClose(tif);
}


// -----------------------------------------------------------------------------
// Load 12-value geometry vectors from geometry file.
// -----------------------------------------------------------------------------

std::vector<double> load_geom_vectors(
    const fs::path& geom_path,
    const std::vector<int>& indices)
{
    std::ifstream ifs(geom_path);

    if (!ifs.is_open()) {
        throw std::runtime_error(
            "Failed to open geometry file: " +
            geom_path.string());
    }


    std::vector<std::array<double, 12>> all_rows;

    std::string line;

    while (std::getline(ifs, line)) {

        if (line.empty()) {
            continue;
        }

        std::istringstream iss(line);

        std::array<double, 12> row{};

        bool valid = true;

        for (int k = 0; k < 12; ++k) {

            if (!(iss >> row[k])) {
                valid = false;
                break;
            }
        }

        if (valid) {
            all_rows.push_back(row);
        }
    }


    std::vector<double> selected;

    selected.reserve(indices.size() * 12);


    for (int idx : indices) {

        if (idx < 0 ||
            static_cast<std::size_t>(idx) >= all_rows.size()) {

            throw std::runtime_error(
                "Geometry index out of range: " +
                std::to_string(idx));
        }


        for (int k = 0; k < 12; ++k) {
            selected.push_back(all_rows[idx][k]);
        }
    }


    return selected;
}


// -----------------------------------------------------------------------------
// Elapsed time.
// -----------------------------------------------------------------------------

double elapsed_sec(
    const std::chrono::steady_clock::time_point& start)
{
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
}

} // namespace


// =============================================================================
// MAIN
// =============================================================================

int main()
{
    try {

        // =====================================================================
        // USER SETTINGS
        // =====================================================================

        const int walnut_id = 1;

        const std::vector<int> orbits_to_recon = {1};

        const int angular_sub_sampling = 10;

        const int voxel_per_mm = 10;


        const fs::path data_path =
            "/content/drive/MyDrive";

        const fs::path recon_path =
            "/content/drive/MyDrive/Walnut1/cppWalnutsOwnReconstructions";


        // =====================================================================
        // TIFF DIMENSIONS
        // =====================================================================

        // Measured directly from the three TIFF files:
        //
        // di000000.tif:  (768, 972)
        // io000000.tif:  (768, 972)
        // scan_000001.tif: (768, 972)
        //
        // No transpose or flip is applied.

        constexpr int projs_rows = 768;
        constexpr int projs_cols = 972;


        if (angular_sub_sampling <= 0) {
            throw std::runtime_error(
                "angular_sub_sampling must be positive");
        }

        if (voxel_per_mm <= 0) {
            throw std::runtime_error(
                "voxel_per_mm must be positive");
        }


        // =====================================================================
        // LOAD AND PREPROCESS PROJECTIONS
        // =====================================================================

        auto t0 = std::chrono::steady_clock::now();

        std::cout << "Load and pre-process data\n";


        const fs::path data_path_full =
            data_path /
            ("Walnut" + std::to_string(walnut_id)) /
            "Projections";


        const std::string dark_name = "di000000.tif";

        const std::vector<std::string> flat_name = {
            "io000000.tif",
            "io000001.tif"
        };

        const std::string vecs_name =
            "scan_geom_corrected.geom";


        // Geometry indices.

        std::vector<int> sample_indices;

        for (int idx = 0; idx < 1200;
             idx += angular_sub_sampling) {

            sample_indices.push_back(idx);
        }


        const int nb_projs_orbit =
            static_cast<int>(sample_indices.size());


        // Projection filenames.
        //
        // This preserves the descending filename order from your
        // original implementation.
        //
        // IMPORTANT:
        // Verify that this descending order corresponds to the
        // selected geometry vectors.

        std::vector<int> projs_idx;

        for (int idx = 1200; idx > 0;
             idx -= angular_sub_sampling) {

            projs_idx.push_back(idx);
        }


        if (projs_idx.size() != sample_indices.size()) {
            throw std::runtime_error(
                "Projection and geometry index counts do not match");
        }


        // Projection memory layout:
        //
        // [detector row, angle, detector column]

        std::vector<float> projs;

        int total_angles = 0;

        std::vector<double> vecs;


        for (int orbit_id : orbits_to_recon) {

            const fs::path orbit_data_path =
                data_path_full /
                ("tubeV" + std::to_string(orbit_id));


            std::cout << "Loading orbit "
                      << orbit_id << "\n";


            // -----------------------------------------------------------------
            // Geometry
            // -----------------------------------------------------------------

            std::vector<double> vecs_orbit =
                load_geom_vectors(
                    orbit_data_path / vecs_name,
                    sample_indices);


            vecs.insert(
                vecs.end(),
                vecs_orbit.begin(),
                vecs_orbit.end());


            // -----------------------------------------------------------------
            // Dark field
            // -----------------------------------------------------------------

            std::vector<float> dark =
                read_tiff(
                    orbit_data_path / dark_name,
                    projs_rows,
                    projs_cols);


            const std::size_t img_elems =
                static_cast<std::size_t>(projs_rows) *
                projs_cols;


            // -----------------------------------------------------------------
            // Flat field
            // -----------------------------------------------------------------

            std::vector<float> flat(img_elems, 0.0f);


            for (const std::string& fn : flat_name) {

                std::vector<float> f_tmp =
                    read_tiff(
                        orbit_data_path / fn,
                        projs_rows,
                        projs_cols);


                for (std::size_t k = 0; k < img_elems; ++k) {
                    flat[k] += f_tmp[k];
                }
            }


            for (std::size_t k = 0; k < img_elems; ++k) {

                flat[k] /=
                    static_cast<float>(flat_name.size());
            }


            // -----------------------------------------------------------------
            // Read and preprocess projections.
            //
            // Temporary layout:
            // [angle, detector row, detector column]
            // -----------------------------------------------------------------

            std::vector<float> projs_orbit(
                static_cast<std::size_t>(nb_projs_orbit) *
                img_elems,
                0.0f);


            for (int i = 0; i < nb_projs_orbit; ++i) {

                std::ostringstream fname;

                fname << "scan_"
                      << std::setw(6)
                      << std::setfill('0')
                      << projs_idx[i]
                      << ".tif";


                std::vector<float> proj_img =
                    read_tiff(
                        orbit_data_path / fname.str(),
                        projs_rows,
                        projs_cols);


                float* dst_slice =
                    projs_orbit.data() +
                    static_cast<std::size_t>(i) * img_elems;


                for (std::size_t k = 0; k < img_elems; ++k) {

                    const float numerator =
                        proj_img[k] - dark[k];

                    const float denominator =
                        flat[k] - dark[k];


                    if (!std::isfinite(numerator) ||
                        !std::isfinite(denominator) ||
                        numerator <= 0.0f ||
                        denominator <= 0.0f) {

                        dst_slice[k] = 0.0f;
                        continue;
                    }


                    const float ratio =
                        numerator / denominator;


                    if (!std::isfinite(ratio) ||
                        ratio <= 0.0f) {

                        dst_slice[k] = 0.0f;
                        continue;
                    }


                    dst_slice[k] = -std::log(ratio);
                }
            }


            // -----------------------------------------------------------------
            // Permute:
            //
            // [angle, row, col]
            //       ->
            // [row, angle, col]
            //
            // Append this orbit along the angle dimension.
            // -----------------------------------------------------------------

            const int new_total_angles =
                total_angles + nb_projs_orbit;


            std::vector<float> new_projs(
                static_cast<std::size_t>(projs_rows) *
                new_total_angles *
                projs_cols,
                0.0f);


            // Copy previously loaded projections.

            for (int r = 0; r < projs_rows; ++r) {

                for (int a = 0; a < total_angles; ++a) {

                    for (int c = 0; c < projs_cols; ++c) {

                        const std::size_t src_i =
                            (static_cast<std::size_t>(r) *
                             total_angles + a) *
                            projs_cols + c;


                        const std::size_t dst_i =
                            (static_cast<std::size_t>(r) *
                             new_total_angles + a) *
                            projs_cols + c;


                        new_projs[dst_i] = projs[src_i];
                    }
                }


                // Copy current orbit.

                for (int a = 0; a < nb_projs_orbit; ++a) {

                    for (int c = 0; c < projs_cols; ++c) {

                        const std::size_t src_i =
                            (static_cast<std::size_t>(a) *
                             projs_rows + r) *
                            projs_cols + c;


                        const std::size_t dst_i =
                            (static_cast<std::size_t>(r) *
                             new_total_angles +
                             (total_angles + a)) *
                            projs_cols + c;


                        new_projs[dst_i] = projs_orbit[src_i];
                    }
                }
            }


            projs = std::move(new_projs);

            total_angles = new_total_angles;
        }


        // ---------------------------------------------------------------------
        // Validate loaded data.
        // ---------------------------------------------------------------------

        if (total_angles <= 0) {
            throw std::runtime_error(
                "No projections were loaded");
        }


        if (vecs.size() !=
            static_cast<std::size_t>(total_angles) * 12) {

            throw std::runtime_error(
                "Geometry vector count does not match projection count");
        }


        std::cout << "Detector dimensions: "
                  << projs_rows << " x " << projs_cols
                  << "\n";

        std::cout << "Loaded "
                  << total_angles
                  << " projections\n";


        std::cout << std::fixed << std::setprecision(3)
                  << elapsed_sec(t0)
                  << " sec elapsed\n";


        // =====================================================================
        // ITERATIVE RECONSTRUCTION
        // =====================================================================

        t0 = std::chrono::steady_clock::now();

        std::cout << "Compute reconstruction:\n";


        const int vol_dim =
            50 * voxel_per_mm + 1;


        const double vox_sz =
            1.0 / static_cast<double>(voxel_per_mm);


        const std::size_t vol_elems =
            static_cast<std::size_t>(vol_dim) *
            vol_dim *
            vol_dim;


        std::vector<float> vol_rec(
            vol_elems,
            0.0f);


        // ---------------------------------------------------------------------
        // ASTRA volume geometry.
        // ---------------------------------------------------------------------

        const float half_extent =
            0.5f * static_cast<float>(vol_dim) *
            static_cast<float>(vox_sz);

        astra::CVolumeGeometry3D scaled_vol_geom(
            vol_dim,
            vol_dim,
            vol_dim,
            -half_extent, -half_extent, -half_extent,  // min X, Y, Z
            half_extent,  half_extent,  half_extent   // max X, Y, Z
        );

        if (!scaled_vol_geom.isInitialized()) {
            throw std::runtime_error(
                "ASTRA volume geometry initialization failed");
        }


        // ---------------------------------------------------------------------
        // Construct ASTRA cone-vector geometry.
        //
        // Geometry file convention:
        //
        // p[0..2]   = source position
        // p[3..5]   = detector center
        // p[6..8]   = detector U vector
        // p[9..11]  = detector V vector
        //
        // SConeProjection.fDetS is the detector corner position.
        // ---------------------------------------------------------------------

        std::vector<astra::SConeProjection> vecs_copy(
            static_cast<std::size_t>(total_angles));


        for (int i = 0; i < total_angles; ++i) {

            const double* p =
                vecs.data() +
                static_cast<std::size_t>(i) * 12;


            astra::SConeProjection& v =
                vecs_copy[static_cast<std::size_t>(i)];


            // Source position.

            v.fSrcX = p[0];
            v.fSrcY = p[1];
            v.fSrcZ = p[2];


            // Detector corner:
            //
            // center - 0.5 * number of columns * U
            //        - 0.5 * number of rows    * V

            v.fDetSX =
                p[3]
                - 0.5 * projs_cols * p[6]
                - 0.5 * projs_rows * p[9];


            v.fDetSY =
                p[4]
                - 0.5 * projs_cols * p[7]
                - 0.5 * projs_rows * p[10];


            v.fDetSZ =
                p[5]
                - 0.5 * projs_cols * p[8]
                - 0.5 * projs_rows * p[11];


            // Detector U vector.

            v.fDetUX = p[6];
            v.fDetUY = p[7];
            v.fDetUZ = p[8];


            // Detector V vector.

            v.fDetVX = p[9];
            v.fDetVY = p[10];
            v.fDetVZ = p[11];
        }


        astra::CConeVecProjectionGeometry3D proj_geom(
            total_angles,
            projs_rows,
            projs_cols,
            std::move(vecs_copy));


        // ---------------------------------------------------------------------
        // Allocate ASTRA data objects.
        // ---------------------------------------------------------------------

        std::unique_ptr<astra::CFloat32VolumeData3D> vol_data(
            astra::createCFloat32VolumeData3DMemory(
                scaled_vol_geom));


        std::unique_ptr<astra::CFloat32ProjectionData3D> sino_data(
            astra::createCFloat32ProjectionData3DMemory(
                proj_geom));


        if (!vol_data || !sino_data) {

            throw std::runtime_error(
                "Failed to allocate ASTRA 3D data objects");
        }


        // Copy measured projections into ASTRA memory.

        std::copy(
            projs.begin(),
            projs.end(),
            sino_data->getFloat32Memory());


        // Initialize reconstruction volume.

        std::copy(
            vol_rec.begin(),
            vol_rec.end(),
            vol_data->getFloat32Memory());


        // ---------------------------------------------------------------------
        // Nesterov reconstruction.
        // ---------------------------------------------------------------------

        ct::NesterovOptions opts;

        opts.num_iterations = 50;

        opts.min_constraint = 0.0f;


        ct::NesterovAcceleratedGradient solver(
            scaled_vol_geom,
            proj_geom,
            *sino_data,
            opts);


        solver.run(*vol_data);


        // Copy reconstructed volume from ASTRA memory.

        std::copy(
            vol_data->getFloat32Memory(),
            vol_data->getFloat32Memory() + vol_elems,
            vol_rec.begin());


        std::cout << std::fixed << std::setprecision(3)
                  << elapsed_sec(t0)
                  << " sec elapsed\n";


        // =====================================================================
        // SAVE RECONSTRUCTION
        // =====================================================================

        t0 = std::chrono::steady_clock::now();

        std::cout << "Save results\n";


        const fs::path recon_path_full =
            recon_path /
            ("Walnut" + std::to_string(walnut_id));


        fs::create_directories(recon_path_full);


        std::string orbit_str = "pos";

        for (int orbit_id : orbits_to_recon) {
            orbit_str += std::to_string(orbit_id);
        }


        const std::size_t slice_elems =
            static_cast<std::size_t>(vol_dim) * vol_dim;


        for (int i = 0; i < vol_dim; ++i) {

            std::ostringstream fname;

            fname << "nnls_"
                  << orbit_str
                  << "_iter" << opts.num_iterations
                  << "_ass" << angular_sub_sampling
                  << "_vmm" << voxel_per_mm
                  << "_"
                  << std::setw(6)
                  << std::setfill('0')
                  << i
                  << ".tiff";


            const float* slice_ptr =
                vol_rec.data() +
                static_cast<std::size_t>(i) * slice_elems;


            write_float_tiff_2d(
                recon_path_full / fname.str(),
                slice_ptr,
                vol_dim,
                vol_dim);
        }


        std::cout << std::fixed << std::setprecision(3)
                  << elapsed_sec(t0)
                  << " sec elapsed\n";


        std::cout << "Reconstruction completed successfully.\n";

    }
    catch (const std::exception& e) {

        std::cerr << "ERROR: "
                  << e.what()
                  << "\n";

        return 1;
    }


    return 0;
}
