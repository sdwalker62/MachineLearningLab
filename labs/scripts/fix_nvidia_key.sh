#!/bin/bash

rm /etc/apt/sources.list.d/cuda.list
rm /etc/apt/sources.list.d/nvidia-ml.list 
apt-get update
sudo sed -i '/developer\.download\.nvidia\.com\/compute\/cuda\/repos/d' /etc/apt/sources.list
sudo apt-key del 7fa2af80
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb