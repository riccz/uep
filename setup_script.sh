#!/bin/bash

set -eu -o pipefail

while true; do
    sudo apt-get update && \
	sudo apt-get install -y \
	     build-essential \
	     cmake \
	     git \
	     libboost-all-dev \
	     libprotobuf-dev \
	     libpthread-stubs0-dev \
	     p7zip \
	     protobuf-compiler \
	     python3-boto3 \
	     python3-dev \
	     python3-matplotlib \
	     python3-numpy \
	     python3-scipy \
	     screen \
	     swig && \
	break
done

git clone --recursive "https://github.com/riccz/uep.git"

pushd uep
git checkout master

mkdir -p build
pushd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(( $(nproc) + 1))

export MPLBACKEND="PDF"
echo 'export MPLBACKEND="PDF"' >> ~/.profile

export PYTHONPATH=$HOME/uep/build/lib:$HOME/uep/src:$HOME/uep
echo 'export PYTHONPATH=$HOME/uep/build/lib:$HOME/uep/src:$HOME/uep' >> ~/.profile

popd
popd
