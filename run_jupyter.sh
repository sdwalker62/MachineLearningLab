#!/bin/sh

sudo docker run --rm -p 8888:8888 -e JUPYTER_ENABLE_LAB=yes \
-v $1/results:/home/jovyan/results \
-v $1/jupyter_notebooks:/home/jovyan/jupyter_notebooks \
-v $1/database:/home/jovyan/database \
jupyter/datascience-notebook:latest
