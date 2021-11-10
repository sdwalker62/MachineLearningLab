# Inspiration for this Dockerfile comes from the gpu-jupyter repository by cschranz. 
# You can find his repository at https://github.com/iot-salzburg/gpu-jupyter 

FROM nvidia/cuda:11.4.2-cudnn8-runtime-ubuntu20.04
LABEL authors="Dalton Walker <samuel.dalton.walker@gmail.com>, Andrew Romans <andrew.romans@hotmail.com>, Michael Fox <>"
LABEL team=SIGIL
LABEL date=2021-11-04



# [0] Base Layer ............. #
RUN chmod 1777 /tmp /var/tmp

ARG NB_USER="jovyan"
ARG NB_UID="1000"
ARG NB_GID="100"

# Fix DL4006
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER root

# Copy a script that we will use to correct permissions after running certain commands
COPY fix-permissions /usr/local/bin/fix-permissions
RUN chmod a+rx /usr/local/bin/fix-permissions

# Instructions located at https://docs.anaconda.com/anaconda/install/linux/
RUN apt-get -q update \
 && apt-get install -yq --no-install-recommends \
    libgl1-mesa-glx \
    libegl1-mesa \
    libxrandr2 \
    libxrandr2 \
    libxss1 \
    libxcursor1 \
    libxcomposite1 \
    libasound2 \
    libxi6 \
    libxtst6 \
    wget \
    ca-certificates \
    sudo \
    locales \
    fonts-firacode \
    fonts-liberation \
    run-one \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

 RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen

#https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
ARG ANACONDA_VERSION="2021.05"
ARG SYSTEM_ARCH="x86_64"
ARG ANACONDA_INSTALLER="Anaconda3-2021.05-Linux-x86_64.sh"
ARG SHA256_HASH="2751ab3d678ff0277ae80f9e8a74f218cfc70fe9a9cdc7bb1c137d7e47e33d53"

