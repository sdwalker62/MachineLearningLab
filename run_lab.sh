#!/bin/bash

docker run \
    --gpus all \
    -p 8888:8888 \
    -e JUPYTER_ENABLE_LAB=1 \
    samuel62/tf_lab:latest