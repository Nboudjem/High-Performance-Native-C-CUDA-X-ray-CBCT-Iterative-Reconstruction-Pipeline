#include <astra/Globals.h>
#include <astra/ConeVecProjectionGeometry3D.h>
#include <astra/VolumeGeometry3D.h>
#include <astra/CudaProjector3D.h>
#include <astra/CudaForwardProjectionAlgorithm3D.h>
#include <astra/CudaBackProjectionAlgorithm3D.h>
#include <astra/Data3D.h>

#include <tiffio.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>
#include <random>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <filesystem>
#include <limits>
#include <cstdint>

namespace fs = std::filesystem;

constexpr int DET_ROWS = 972;
constexpr int DET_COLS = 768;
constexpr int TOTAL_VIEWS = 1200;
constexpr int SUBSAMPLING = 10;
constexpr int ANGLES = 120;

constexpr int VOL_SIZE = 501;
constexpr size_t VOL_ELEMENTS =
    size_t(VOL_SIZE) * VOL_SIZE * VOL_SIZE;

constexpr size_t SINO_ELEMENTS =
    size_t(DET_COLS) * ANGLES * DET_ROWS;

const std::string BASE =
    "/kaggle/input/datasets/boudjemianacer/cbct-data/"
    "Walnut1/Projections/tubeV1/";

const std::string OUTPUT =
    "/kaggle/working/WalnutsOwnReconstructions/"
    "Walnut1/";

const std::string GEOMETRY_FILE =
    BASE + "scan_geom_corrected.geom";

const int GPU_ID = 0;


// ============================================================
// TIFF READER
// ============================================================

std::vector<uint16_t> readTiff(const std::string& filename)
{
    TIFF* tif = TIFFOpen(filename.c_str(), "r");

    if (!tif)
        throw std::runtime_error("Cannot open TIFF: " + filename);

    uint32_t width = 0;
    uint32_t height = 0;

    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);

    if (width != DET_ROWS || height != DET_COLS) {
        TIFFClose(tif);
        throw std::runtime_error(
            "Unexpected TIFF dimensions in " + filename +
            ": width=" + std::to_string(width) +
            ", height=" + std::to_string(height));
    }

    uint16_t bits = 0;
    uint16_t samples = 0;
    uint16_t format = SAMPLEFORMAT_UINT;

    TIFFGetFieldDefaulted(tif, TIFFTAG_BITSPERSAMPLE, &bits);
    TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLESPERPIXEL, &samples);
    TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLEFORMAT, &format);

    if (bits != 16 || samples != 1 ||
        format != SAMPLEFORMAT_UINT) {
        TIFFClose(tif);
        throw std::runtime_error(
            "Expected single-channel uint16 TIFF: " + filename);
    }

    std::vector<uint16_t> raw(size_t(width) * height);

    for (uint32_t y = 0; y < height; ++y) {
        if (TIFFReadScanline(
                tif, raw.data() + size_t(y) * width, y, 0) < 0) {
            TIFFClose(tif);
            throw std::runtime_error(
                "TIFF scanline read failed: " + filename);
        }
    }

    TIFFClose(tif);
    return raw;
}


// ============================================================
// IMAGE ORIENTATION
//
// Python:
//     np.transpose(np.flipud(image))
//
// Original raw shape: (768, 972)
// Transformed shape:  (972, 768)
//
// transformed[r,c] = raw[767-c,r]
// ============================================================

std::vector<float> readTransformed(
    const std::string& filename)
{
    std::vector<uint16_t> raw = readTiff(filename);

    std::vector<float> result(
        size_t(DET_ROWS) * DET_COLS);

    for (int r = 0; r < DET_ROWS; ++r) {
        for (int c = 0; c < DET_COLS; ++c) {
            result[size_t(r) * DET_COLS + c] =
                static_cast<float>(
                    raw[size_t(DET_COLS - 1 - c) * DET_ROWS + r]);
        }
    }

    return result;
}


// ============================================================
// LOAD DARK AND FLAT FIELD
// ============================================================

void loadCalibration(
    std::vector<float>& dark,
    std::vector<float>& flat)
{
    std::cout << "Loading dark image...\n";

    dark = readTransformed(BASE + "di000000.tif");

    std::cout << "Loading flat images...\n";

    std::vector<float> flat0 =
        readTransformed(BASE + "io000000.tif");

    std::vector<float> flat1 =
        readTransformed(BASE + "io000001.tif");

    flat.resize(dark.size());

    for (size_t i = 0; i < flat.size(); ++i) {
        flat[i] = 0.5f * (flat0[i] + flat1[i]);
    }

    std::cout << "Calibration loaded.\n";
}


// ============================================================
// LOAD MEASUREMENTS
//
// Projection file order from the Python main script:
//
//     range(1200, 0, -10)
//
// Therefore:
//
//     scan_001200.tif
//     scan_001190.tif
//     ...
//     scan_000010.tif
//
// The geometry uses the first 1200 rows, sampled 0,10,...1190.
// ============================================================

void loadProjections(
    const std::vector<float>& dark,
    const std::vector<float>& flat,
    std::vector<float>& sino)
{
    sino.resize(SINO_ELEMENTS);

    std::cout << "Loading 120 projections...\n";

    for (int angle = 0; angle < ANGLES; ++angle) {

        const int projectionNumber =
            TOTAL_VIEWS - angle * SUBSAMPLING;

        std::ostringstream name;
        name << BASE << "scan_"
             << std::setw(6) << std::setfill('0')
             << projectionNumber << ".tif";

        std::cout << "\rProjection "
                  << angle + 1 << "/" << ANGLES
                  << std::flush;

        std::vector<float> projection =
            readTransformed(name.str());

        for (size_t pixel = 0;
             pixel < projection.size(); ++pixel) {

            const float numerator =
                projection[pixel] - dark[pixel];

            const float denominator =
                flat[pixel] - dark[pixel];

            if (!(numerator > 0.0f) ||
                !(denominator > 0.0f)) {
                throw std::runtime_error(
                    "Non-positive intensity during "
                    "dark/flat correction.");
            }

            const float ratio = numerator / denominator;

            const float value = -std::log(ratio);

            if (!std::isfinite(value)) {
                throw std::runtime_error(
                    "Non-finite projection value.");
            }

            // ASTRA layout: (detector column, angle, detector row)
            // Same flat memory ordering as Python array
            // after transpose from (rows, angles, cols).
            sino[size_t(angle) * DET_ROWS * DET_COLS + pixel] =
                value;
        }
    }

    std::cout << "\nProjection preprocessing complete.\n";
}


// ============================================================
// LOAD CONE-VECTOR GEOMETRY
// ============================================================

astra::CConeVecProjectionGeometry3D loadGeometry()
{
    std::ifstream file(GEOMETRY_FILE);

    if (!file)
        throw std::runtime_error(
            "Cannot open geometry file: " + GEOMETRY_FILE);

    std::vector<astra::SConeProjection> vectors;
    vectors.reserve(ANGLES);

    std::string line;
    int row = 0;

    while (row < TOTAL_VIEWS && std::getline(file, line)) {

        std::istringstream stream(line);
        double d[12];

        for (int j = 0; j < 12; ++j) {
            if (!(stream >> d[j])) {
                throw std::runtime_error(
                    "Invalid geometry row " +
                    std::to_string(row));
            }
        }

        if (row % SUBSAMPLING == 0) {

            astra::SConeProjection p;

            p.fSrcX = d[0];
            p.fSrcY = d[1];
            p.fSrcZ = d[2];

            p.fDetUX = d[6];
            p.fDetUY = d[7];
            p.fDetUZ = d[8];

            p.fDetVX = d[9];
            p.fDetVY = d[10];
            p.fDetVZ = d[11];

            // Geometry file specifies detector center.
            // ASTRA cone-vector geometry expects bottom-left corner.

            p.fDetSX =
                d[3]
                - 0.5 * DET_ROWS * p.fDetVX
                - 0.5 * DET_COLS * p.fDetUX;

            p.fDetSY =
                d[4]
                - 0.5 * DET_ROWS * p.fDetVY
                - 0.5 * DET_COLS * p.fDetUY;

            p.fDetSZ =
                d[5]
                - 0.5 * DET_ROWS * p.fDetVZ
                - 0.5 * DET_COLS * p.fDetUZ;

            vectors.push_back(p);
        }

        ++row;
    }

    if (row != TOTAL_VIEWS || vectors.size() != ANGLES)
        throw std::runtime_error(
            "Incorrect number of geometry rows or views.");

    std::cout << "Geometry loaded: "
              << vectors.size() << " views.\n";

    return astra::CConeVecProjectionGeometry3D(
        ANGLES, DET_ROWS, DET_COLS, std::move(vectors));
}


// ============================================================
// SIMPLE NUMERICAL HELPERS
// ============================================================

double norm2(const float* data, size_t n)
{
    double sum = 0.0;

    for (size_t i = 0; i < n; ++i)
        sum += double(data[i]) * data[i];

    return std::sqrt(sum);
}

void copyArray(
    const float* src,
    float* dst,
    size_t n)
{
    std::copy(src, src + n, dst);
}

void zeroArray(float* data, size_t n)
{
    std::fill(data, data + n, 0.0f);
}


// ============================================================
// TIFF OUTPUT
// ============================================================

void saveVolume(
    const float* volume,
    int iteration)
{
    fs::create_directories(OUTPUT);

    std::cout << "\nSaving reconstructed slices...\n";

    for (int z = 0; z < VOL_SIZE; ++z) {

        std::ostringstream filename;

        filename << OUTPUT
                 << "nnls_pos1_iter" << iteration
                 << "_ass10_vmm10_"
                 << std::setw(6) << std::setfill('0')
                 << z << ".tiff";

        TIFF* tif = TIFFOpen(filename.str().c_str(), "w");

        if (!tif)
            throw std::runtime_error(
                "Cannot create output TIFF: " + filename.str());

        TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, VOL_SIZE);
        TIFFSetField(tif, TIFFTAG_IMAGELENGTH, VOL_SIZE);
        TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
        TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 32);
        TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
        TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
        TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
        TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
        TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_NONE);

        const float* slice =
            volume + size_t(z) * VOL_SIZE * VOL_SIZE;

        for (int y = 0; y < VOL_SIZE; ++y) {

            void* row = const_cast<float*>(
                slice + size_t(y) * VOL_SIZE);

            if (TIFFWriteScanline(tif, row, y, 0) < 0) {
                TIFFClose(tif);
                throw std::runtime_error(
                    "Failed writing output TIFF.");
            }
        }

        TIFFClose(tif);
    }

    std::cout << "Saved " << VOL_SIZE
              << " slices in:\n" << OUTPUT << "\n";
}


// ============================================================
// MAIN RECONSTRUCTION
// ============================================================

int main()
{
    try {

        std::cout << std::setprecision(9);

        std::cout << "====================================\n";
        std::cout << "ASTRA CUDA APGD RECONSTRUCTION\n";
        std::cout << "Walnut 1 - tubeV1\n";
        std::cout << "====================================\n";

        if (!astra::cudaAvailable())
            throw std::runtime_error("ASTRA CUDA unavailable.");

        // ----------------------------------------------------
        // 1. Geometry
        // ----------------------------------------------------

        auto projectionGeometry = loadGeometry();

        astra::CVolumeGeometry3D volumeGeometry(
            VOL_SIZE, VOL_SIZE, VOL_SIZE,
            -25.05f, -25.05f, -25.05f,
             25.05f,  25.05f,  25.05f);

        astra::CCudaProjector3D projector(
            projectionGeometry, volumeGeometry);

        // ----------------------------------------------------
        // 2. Load and preprocess projection measurements
        // ----------------------------------------------------

        std::vector<float> dark;
        std::vector<float> flat;
        std::vector<float> measured;

        loadCalibration(dark, flat);
        loadProjections(dark, flat, measured);

        dark.clear();
        dark.shrink_to_fit();

        flat.clear();
        flat.shrink_to_fit();

        // ----------------------------------------------------
        // 3. Allocate ASTRA memory
        // ----------------------------------------------------

        std::unique_ptr<astra::CFloat32ProjectionData3D>
            measuredData(
                astra::createCFloat32ProjectionData3DMemory(
                    projectionGeometry));

        std::unique_ptr<astra::CFloat32ProjectionData3D>
            predictedData(
                astra::createCFloat32ProjectionData3DMemory(
                    projectionGeometry));

        std::unique_ptr<astra::CFloat32VolumeData3D>
            xData(
                astra::createCFloat32VolumeData3DMemory(
                    volumeGeometry));

        std::unique_ptr<astra::CFloat32VolumeData3D>
            bpData(
                astra::createCFloat32VolumeData3DMemory(
                    volumeGeometry));

        if (!measuredData || !predictedData ||
            !xData || !bpData)
            throw std::runtime_error(
                "ASTRA memory allocation failed.");

        float* s = measuredData->getFloat32Memory();
        float* predicted = predictedData->getFloat32Memory();
        float* x = xData->getFloat32Memory();
        float* bp = bpData->getFloat32Memory();

        if (!s || !predicted || !x || !bp)
            throw std::runtime_error(
                "ASTRA float32 memory unavailable.");

        copyArray(measured.data(), s, SINO_ELEMENTS);
        zeroArray(predicted, SINO_ELEMENTS);

        zeroArray(x, VOL_ELEMENTS);
        zeroArray(bp, VOL_ELEMENTS);

        measured.clear();
        measured.shrink_to_fit();

        std::cout << "\nASTRA buffers allocated.\n";

        // ----------------------------------------------------
        // 4. Initialize CUDA algorithms
        // ----------------------------------------------------

        astra::CCudaForwardProjectionAlgorithm3D fp;

        if (!fp.initialize(
                &projector, predictedData.get(),
                xData.get(), {GPU_ID}, -1))
            throw std::runtime_error(
                "Forward projection initialization failed.");

        // BP for measured data: ATy = A^T s
        astra::CCudaBackProjectionAlgorithm3D bpMeasured;

        if (!bpMeasured.initialize(
                &projector, measuredData.get(), bpData.get()))
            throw std::runtime_error(
                "Measured backprojection initialization failed.");

        bpMeasured.setGPUIndices({GPU_ID});

        // BP for predicted data: A^T A x
        astra::CCudaBackProjectionAlgorithm3D bpPredicted;

        if (!bpPredicted.initialize(
                &projector, predictedData.get(), bpData.get()))
            throw std::runtime_error(
                "Predicted backprojection initialization failed.");

        bpPredicted.setGPUIndices({GPU_ID});

        // ----------------------------------------------------
        // 5. Compute ATy = A^T s
        // ----------------------------------------------------

        std::vector<float> ATy(VOL_ELEMENTS);

        std::cout << "\nComputing A^T s...\n";

        zeroArray(bp, VOL_ELEMENTS);

        if (!bpMeasured.run())
            throw std::runtime_error(
                "Backprojection of measurements failed.");

        copyArray(bp, ATy.data(), VOL_ELEMENTS);

        // ----------------------------------------------------
        // 6. Power iteration
        //
        // Estimate spectral norm of A^T A.
        // Equivalent to Python:
        //     b_k1 = A.T * A * b_k
        //     b_k1_norm = norm(b_k1)
        //     b_k = b_k1 / b_k1_norm
        // ----------------------------------------------------

        std::cout << "\nRunning 10 power iterations...\n";

        std::vector<float> randomVolume(VOL_ELEMENTS);

        std::mt19937 generator(12345);
        std::uniform_real_distribution<float> distribution(
            0.0f, 1.0f);

        for (float& value : randomVolume)
            value = distribution(generator);

        double lipschitz = 0.0;

        for (int k = 0; k < 10; ++k) {

            copyArray(randomVolume.data(), x, VOL_ELEMENTS);

            zeroArray(predicted, SINO_ELEMENTS);

            if (!fp.run())
                throw std::runtime_error(
                    "Power iteration FP failed.");

            zeroArray(bp, VOL_ELEMENTS);

            if (!bpPredicted.run())
                throw std::runtime_error(
                    "Power iteration BP failed.");

            lipschitz = norm2(bp, VOL_ELEMENTS);

            if (!(lipschitz > 0.0) ||
                !std::isfinite(lipschitz))
                throw std::runtime_error(
                    "Invalid Lipschitz estimate.");

            for (size_t i = 0; i < VOL_ELEMENTS; ++i)
                randomVolume[i] =
                    static_cast<float>(bp[i] / lipschitz);

            std::cout << "Power iteration "
                      << k + 1 << "/10: "
                      << lipschitz << "\n";
        }

        const float nu = static_cast<float>(1.0 / lipschitz);

        std::cout << "Estimated Lipschitz constant: "
                  << lipschitz << "\n";

        std::cout << "Step size nu: " << nu << "\n";

        randomVolume.clear();
        randomVolume.shrink_to_fit();

        // ----------------------------------------------------
        // 7. APGD initialization
        //
        // Preserve Python plugin initialization:
        // x = initial volume (zero)
        // x_old = x
        // NRMx = zero
        // NRMx_old = zero
        // gradient = -ATy
        // ----------------------------------------------------

        zeroArray(x, VOL_ELEMENTS);

        std::vector<float> x_old(VOL_ELEMENTS, 0.0f);
        std::vector<float> NRMx(VOL_ELEMENTS, 0.0f);
        std::vector<float> NRMx_old(VOL_ELEMENTS, 0.0f);
        std::vector<float> gradient(VOL_ELEMENTS);

        for (size_t i = 0; i < VOL_ELEMENTS; ++i)
            gradient[i] = -ATy[i];

        double t_acc = 1.0;
        double bestObjective =
            std::numeric_limits<double>::infinity();

        constexpr int ITERATIONS = 100;

        std::cout << "\n====================================\n";
        std::cout << "Starting APGD: "
                  << ITERATIONS << " iterations\n";
        std::cout << "====================================\n";

        // ----------------------------------------------------
        // 8. APGD iterations
        // ----------------------------------------------------

        for (int iteration = 0;
             iteration < ITERATIONS; ++iteration) {

            const double tau =
                (t_acc - 1.0) / (t_acc + 2.0);

            t_acc += 1.0;

            // Descent direction:
            //
            // gradient
            // - tau/nu * (x - x_old)
            // + tau * (NRMx - NRMx_old)

            for (size_t i = 0; i < VOL_ELEMENTS; ++i) {

                const double descent =
                    gradient[i]
                    - (tau / nu) * (x[i] - x_old[i])
                    + tau * (NRMx[i] - NRMx_old[i]);

                x_old[i] = x[i];

                const float updated =
                    static_cast<float>(x[i] - nu * descent);

                // MinConstraint = 0
                // MaxConstraint = None
                x[i] = std::max(0.0f, updated);
            }

            // Wx = A x
            zeroArray(predicted, SINO_ELEMENTS);

            if (!fp.run())
                throw std::runtime_error(
                    "FP failed at iteration " +
                    std::to_string(iteration + 1));

            // Objective = 0.5 * ||Wx - s||^2

            double objective = 0.0;

            for (size_t i = 0; i < SINO_ELEMENTS; ++i) {
                const double residual =
                    double(predicted[i]) - s[i];

                objective += residual * residual;
            }

            objective *= 0.5;

            // NRMx = A^T A x
            zeroArray(bp, VOL_ELEMENTS);

            if (!bpPredicted.run())
                throw std::runtime_error(
                    "BP failed at iteration " +
                    std::to_string(iteration + 1));

            // Preserve previous NRMx.
            copyArray(
                NRMx.data(),
                NRMx_old.data(),
                VOL_ELEMENTS);

            // Copy BP result to NRMx.
            copyArray(
                bp,
                NRMx.data(),
                VOL_ELEMENTS);

            // gradient = NRMx - ATy
            for (size_t i = 0; i < VOL_ELEMENTS; ++i)
                gradient[i] = NRMx[i] - ATy[i];

            // Restart condition from Python:
            // objective[i] > min(objective[0:i+1])
            //
            // This is equivalent to comparing with the
            // minimum objective from previous iterations.

            if (objective > bestObjective) {
                t_acc = 1.0;
            }

            bestObjective = std::min(
                bestObjective, objective);

            std::cout
                << "Iteration "
                << std::setw(3) << iteration + 1
                << "/" << ITERATIONS
                << " | Objective: "
                << std::scientific << objective
                << " | Best: " << bestObjective
                << " | t: " << t_acc
                << std::defaultfloat
                << "\n";
        }

        // ----------------------------------------------------
        // 9. Save volume
        // ----------------------------------------------------

        saveVolume(x, ITERATIONS);

        std::cout << "\n====================================\n";
        std::cout << "RECONSTRUCTION COMPLETED\n";
        std::cout << "====================================\n";

        return 0;
    }
    catch (const std::exception& error) {

        std::cerr << "\nFATAL ERROR: "
                  << error.what() << "\n";

        return 1;
    }
}

