#!/bin/bash

## Update system and install the necessary libraries:
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip libopenblas-dev libjpeg-dev zlib1g-dev libavcodec-dev libavformat-dev libswscale-dev
pip3 install --upgrade pip

##Install cuSPARSELt:
CUSPARSELT_VER="0.7.1.0"
wget https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-aarch64/libcusparse_lt-linux-aarch64-${CUSPARSELT_VER}-archive.tar.xz
tar xf *.tar.xz
sudo cp -a */include/* /usr/local/cuda/include/
sudo cp -a */lib/* /usr/local/cuda/lib64/
sudo ldconfig

##Install cuDSS
wget https://developer.download.nvidia.com/compute/cudss/0.7.1/local_installers/cudss-local-tegra-repo-ubuntu2204-0.7.1_0.7.1-1_arm64.deb
sudo dpkg -i cudss-local-tegra-repo-ubuntu2204-0.7.1_0.7.1-1_arm64.deb
sudo cp /var/cudss-local-tegra-repo-ubuntu2204-0.7.1/cudss-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudss

## Installing Pytorch from Jetson AI Lab Community PyPI Mirror
## Ref link https://pypi.jetson-ai-lab.io/

##Pytorch
pip3 install "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/02f/de421eabbf626/torch-2.9.1-cp310-cp310-linux_aarch64.whl#sha256=02fde421eabbf62633092de30405ea4d917323c55bea22bfd10dfeb1f1023506"

##Torch Audio
pip3 install "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/d12/bede7113e6b00/torchaudio-2.9.1-cp310-cp310-linux_aarch64.whl#sha256=d12bede7113e6b00f7c5ed53a28f7fa44a624780c8097a6a2352f32548d77ffb"

##Torch Vision
pip3 install "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/d5b/caaf709f11750/torchvision-0.24.1-cp310-cp310-linux_aarch64.whl#sha256=d5bcaaf709f11750b5bb0f6ec30f37605da2f3d5cb3cd2b0fe5fac2850e08642"

##Clean up
rm -r libcusparse_lt-linux-aarch64-0.7.1.0-archive.tar.xz libcusparse_lt-linux-aarch64-0.7.1.0-archive cudss-local-tegra-repo-ubuntu2204-0.7.1_0.7.1-1_arm64.deb

##Verify Installs
python3 - <<EOF
import torch, torchvision
print("Torch      :", torch.__version__)
print("TorchVision:", torchvision.__version__)
print("CUDA avail?:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU device :", torch.cuda.get_device_name(0))
EOF
