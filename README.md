# log-analyzer

## Introduction

This repository is the main project page for research into log messages and their potential for determining system state information. The following directories, listed and defined in no particular order, comprise the project: 

jupyter: used to build the docker container. (section 1)
notebook: This directory will contains the necessary code to standup the jupyter notebook. (section 2)
data: All databases and other source data will be stored here, note this is not where models and other serialized derived data is to be stored.
reults: Data which is derivative of information from the data directory will be kept here. (section 3)
doc: Any useful documentation will be kept in this directory. This includes research papers and notes. 
preprocessing: The code for the pre-processing pipeline. (section 4)
training: The transformer code is maintained here. Note this is not where saved models are stored, check the results folder. (section 5)

## 0. First Steps 

Hey there! In this section I will describe the steps to getting the jupyter container running. If you are experienced and just want the tldr then here it is: make sure there are database files in the data dir (message me if you need these files) and then stand up the jupyter container using docker-compose (use the --build flag: sudo docker-compose up --build jupyter). Below is the gentle introduction.

## 1. jupyter 

This is the main directory for experimentation. It houses all the code necessary to build the jupyter docker container. The notebooks, results, data, and doc directories are mapped to this container through docker-compose for serializing/deserializing objects, loading trained models, and modifying LaTeX reports. 

## 2. notebooks 

This is source folder for all Jupyter notebooks. As Jupyter is the primary tool for experimentation this folder houses all experimental code. Updates and modifications to the pre-processing and training folders more than likely originate from this directory. Currently there is only the longruntransformer.ipynb notebook in this directory which holds the experimental transformer code. I will soon add a playground.ipynb notebook for throwaway code. 

## 3. results 

All trained models, pickeled intermediary results, and tensorflow checkpoints, and static graphs will be kept here. This directory is mapped to the jupyter container through docker-compose for loading and saving objects. 

## 4. preprocessing 

## 5. training
