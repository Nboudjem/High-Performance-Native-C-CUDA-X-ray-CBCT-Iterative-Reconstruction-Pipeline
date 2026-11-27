## High-Performance-Native-C-CUDA-X-ray-CBCT-Iterative-Reconstruction-Pipeline
## Under optimization and tests !!!!!
A high-performance native C++ and CUDA implementation for 3D X-ray Computed Tomography (CT) Iterative reconstruction, optimized for the **Walnut dataset**. This project migrates a Python/ASTRA-toolbox workflow into a streamlined, high-throughput GPU architecture utilizing custom Nesterov Accelerated Gradient Descent (AGD) and OpenCV preprocessing.


---

## Execute the code
g++ -std=c++17 -O3 -DASTRA_CUDA \
main.cpp NesterovOptimizer.cpp \
-I/content/astra-toolbox/include \
-I/content/astra-toolbox/include/astra \
-L/content/astra-toolbox/build/linux/.libs \
-Wl,-rpath,/content/astra-toolbox/build/linux/.libs \
-lastra -ltiff -o main

export LD_LIBRARY_PATH=/content/astra-toolbox/build/linux/.libs:$LD_LIBRARY_PATH
./main

## 🚀 Key Features

* **Native C++17 & CUDA 13.0 Pipeline**: Eliminates Python interpreter overhead during heavy tensor operations.
* **GPU-Accelerated Nesterov AGD**: Custom iterative optimization loop running entirely on device memory.
* **Robust Preprocessing**: Built-in dark/flat-field calibration, transposition, and Beer-Lambert law linearization using OpenCV.
* **Dynamic 16-bit TIFF Export**: Automatic dynamic-range contrast scaling and export of volumetric slices into 16-bit unsigned integer TIFF stacks compatible with ImageJ, Fiji, and Python analysis pipelines.

---

## 🛠️ Toolchain & Dependencies

* **Compiler**: GCC 13+ (with `-fpermissive` template compatibility workarounds for legacy headers)
* **Build System**: CMake 3.20+
* **CUDA Toolkit**: Version 13.0 (Tested on NVIDIA Tesla T4)
* **OpenCV**: Version 4.6.0+
* **ASTRA-toolbox**: C++ development headers and libraries installed at `/usr/local/astra`

---

## 📂 Project Structure

```text
WalnutCBCT/
├── CMakeLists.txt        # Root CMake configuration
├── include/              # Header files (e.g., NesterovOptimizer.hpp)
├── src/                  # Source files
│   ├── main.cpp          # Pipeline orchestration, I/O, and slice export
│   └── NesterovOptimizer.cpp # CUDA optimization kernels & loops
└── build/                # Build output directory


---

## 📥 Installing ASTRA-Toolbox for C++

To enable native C++ and CUDA projection operations, you need to clone and build the ASTRA-toolbox from source so its headers and libraries are available to CMake.

```bash
# 1. Clone the ASTRA-toolbox repository
git clone https://github.com/astra-toolbox/astra-toolbox.git
cd astra-toolbox

# 2. Configure and build (adjust CUDA paths if necessary)
./autogen.sh
./configure --with-cuda=/usr/local/cuda
make -j$(nproc)

# 3. Install headers and libraries system-wide (installs to /usr/local/astra)
sudo make install
```

---

## ⚙️ Building the Project

1. **Clone your repository and navigate into it**:
   ```bash
   git clone https://github.com/your-username/WalnutCBCT-CUDA.git
   cd WalnutCBCT-CUDA
   ```

2. **Create and enter the build directory**:
   ```bash
   mkdir -p build && cd build
   ```

3. **Configure the project with CMake**:
   ```bash
   cmake -DCMAKE_BUILD_TYPE=Release ..
   ```

4. **Compile the executable**:
   ```bash
   make -j$(nproc)
   ```

---

## 🏃 Execution

Run the compiled executable directly from your build directory:

```bash
./WalnutCT
```

The pipeline will:
1. Load dark fields, flat fields, and projection TIFF stacks.
2. Apply Beer-Lambert linearization and build the projection tensor buffer.
3. Initialize GPU memory buffers and execute the Nesterov AGD optimization loop.
4. Scale and export the reconstructed volume as a stack of 16-bit TIFF slices to your designated output directory.

---

