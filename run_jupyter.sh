#!/bin/sh

sudo docker run --rm -p 8888:8888 -e JUPYTER_ENABLE_LAB=yes -v /Users/dalton/Documents/Work/ClusterAnalysisTool/results:/home/jovyan/results -v /Users/dalton/Documents/Work/ClusterAnalysisTool/training:/home/jovyan/work jupyter/datascience-notebook:latest

