#!/bin/bash
# filename: xnnpack_mode.sh

cp CMakeLists_XNNPACK.txt ./et-nanogpt/CMakeLists.txt
cp export_nanogpt_xnnpack.py ./et-nanogpt/export_nanogpt.py

cd et-nanogpt/

(rm -rf cmake-out && mkdir cmake-out && cd cmake-out && cmake ..)
cmake --build cmake-out -j$(nproc)

cd ../
