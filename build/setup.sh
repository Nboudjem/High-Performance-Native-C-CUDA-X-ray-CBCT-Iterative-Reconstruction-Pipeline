#!/bin/bash
set -e

echo "=== Updating system & installing dependencies ==="
apt-get update -y
apt-get install -y build-essential cmake git libopencv-dev libtool autoconf automake

echo "=== Setting up ASTRA-toolbox ==="
if [ ! -d "/content/astra-toolbox" ]; then
    git clone https://github.com/astra-toolbox/astra-toolbox.git /content/astra-toolbox
fi

cd /content/astra-toolbox/build/linux
./autogen.sh
./configure --with-cuda=/usr/local/cuda --prefix=/usr/local/astra
make -j$(nproc)
make install

echo "=== Environment restored successfully! ==="
