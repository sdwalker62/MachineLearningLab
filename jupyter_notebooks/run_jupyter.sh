#!/bin/sh

sudo docker run --rm -p 8888:8888 -e JUPYTER_ENABLE_LAB=yes \
-v /home/aromans/Dev/log-analyzer/results:/home/jovyan/results \
-v /home/aromans/Dev/log-analyzer/jupyter_notebooks:/home/jovyan/work \
-v /home/aromans/Dev/log-analyzer/database:/home/jovyan/database \
jupyter/datascience-notebook:latest
