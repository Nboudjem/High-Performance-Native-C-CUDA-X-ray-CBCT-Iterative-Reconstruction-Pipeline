#!/bin/bash
set -e

echo "========================================"
echo " Building & Executing WalnutCT Pipeline"
echo "========================================"

PROJECT_ROOT="/content/drive/MyDrive/WalnutCBCT"
BUILD_DIR="$PROJECT_ROOT/build"

if [ ! -d "$BUILD_DIR" ]; then
    echo "[BUILD] Creating build directory..."
    mkdir -p "$BUILD_DIR"
fi

cd "$BUILD_DIR"

echo "[BUILD] Running CMake..."
cmake -DCMAKE_BUILD_TYPE=Release ..

echo "[BUILD] Compiling project..."
make -j$(nproc)

echo "========================================"
echo " Compilation Complete! Starting Pipeline..."
echo "========================================"

./WalnutCT

echo "========================================"
echo " Pipeline execution finished successfully!"
echo "========================================"
