# log-analyzer

### Table of Contents
<!-- TOC -->
- [Introduction](#introduction)
- [First Steps](#first-steps)
  - [Install Docker](#install-docker-engine)
  - [Install Docker-Compose](#install-docker-compose)
  - [Run the Jupyter Container](#run-the-jupyter-container)
- [Jupyter](#jupyter)
- [Notebooks](#notebooks)
- [Docs](#docs)
- [Pre-processing](#pre-processing)
- [Training](#training)
<!-- /TOC -->

## Introduction

This repository is the main project page for research into log messages and their potential for determining system state information. The following directories, listed and defined in no particular order, comprise the project: 

* jupyter: The docker container for our jupyterlab environment. 
* notebook: Repository for our jupyter notebooks, only contains \*.ipynb files  
* data: All databases and other source data will be stored here, note this is not where models and other serialized derived data is to be stored.
* reults: Data which is derivative of information from the data directory will be kept here.
* doc: Any useful documentation will be kept in this directory. This includes research papers and notes. 
* preprocessing: The code for the pre-processing pipeline. 
* training: The transformer code is maintained here. Note this is not where saved models are stored, check the results folder. 

## First Steps 

Hey there! In this section I will describe the steps to getting the jupyter container running. If you are experienced and just want the tldr then here it is: make sure there are database files in the data dir (message me if you need these files) and then stand up the jupyter container using docker-compose (use the --build flag: sudo docker-compose up --build jupyter). Below is the gentle introduction.

I highly recommend setting up ssh for GitHub. The use of passwords will soon be depreciated. Here are some links on how to do this. Note you will need to setup an ssh key for each computer you wish to use and the key is repository agnostic (meaning you will not need a new key for each repository you access). 

* Some information on SSH: https://www.hostinger.com/tutorials/ssh-tutorial-how-does-ssh-work
* GitHub's guide to setting up an ssh key for your machine: https://docs.github.com/en/enterprise-server@3.0/github/authenticating-to-github/adding-a-new-ssh-key-to-your-github-account

I also highly recommend using some flavor of Linux. Docker is heavily used in this project and runs better on Linux since it does not require a VM unlike macOS and Windows 10. Although this is highly controversial I would recommend using Ubuntu 21.04 or 20.04 LTS. If Cannonical disgust you then I would recommend using MX Linux or Manjoro. 

All instructions in this document will assume the use of linux, if you are using WSL2 or macOS there may be slight alterations needed to make the code work. I will write about this at the end of the section. Also at the end will be a quick introduction on setting up Docker on WSL2 (Windows) and macOS. If you need further help please message me and I will try to help where I can. 

Please note that if you intend to use CUDA to leverage your GPU then linux is a requirement. I know that GPU passthrough is a thing on the Window's developer ring and thus technically useable with WSL2 using Nvidia's beta drivers. I would advise against this route unless you truly know what you are doing and are willing to accept the risk of an unstable Windows 10. 

From here on I will assume you are using Ubuntu and thus aptitude. If you are using yum or pacman just substitute those commands or using Docker's documentation for installing on those distributions. 

First a word on why we use Docker. I will try to explain our rationale for the tools we use whenever appropriate. Of course some of this is a matter of opinion.

Docker allows for us to easily package and manage our environments. The members of our team use Linux (Arch/Ubuntu), macOS, and Windows 10. To provide a more consistent experience across all of these platforms we package our environment using Docker containers. We can design these containers however we would like and using whatever verions are necessary. Instead of requiring all members to use Python 3.9, which many aren't using and would require then to setup yet *another* environment using pyenv or virtualenv, we simply record that decision in the Dockerfile. Since we do not move large files from the host machine to the Docker VM (if not on Linux) we don't experience any performance degradation when using Docker, so from our perspective there are only positives. I will talk about the design of our container later.

Now let's get started. 

### Install Docker Engine

To start you will need docker. Check to see if any previous version of Docker is lingering around - we will need to remove it if so. 

```console
sudo apt-get remove docker docker-engine docker.io containerd runc
```

Don't worry if apt-get reports that none of these packages are installed. Next we will setup the Docker repository:


```console
 sudo apt-get update
 sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release
```

Now we will add Docker's official GPG (GNU Privacy Guard) key. This is necessary as it will sign the code as being legitimate and thus enabling encrypted information to be transferred.

```console
 curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
```
We will use the **stable** repositry. If you wish to use the nightly or test channels then replace the word stable in the above command. Note that the below command is for x86_64 / amd64 only! If you are using arm64 or armhf refer to the documentation here: https://docs.docker.com/engine/install/ubuntu/

```console
echo \
  "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

Now that we have the signed repo setup we can install Docker: 

```console 
 sudo apt-get update
 sudo apt-get install docker-ce docker-ce-cli containerd.io
```
Note that this will install the *latest* docker engine. If you need a specific version for compatability reasons please refer to the documentation. 

Finally we will check that the installation was successful:

```console
sudo docker run hello-world
```

### Install Docker-Compose 

We don't just have a singular Docker container. If that was true we could use the docker run command for our needs. Instead we have specialized Docker containers that each are responsible for a single task. Our design philosophy follows the Single Responsibility Principle. https://en.wikipedia.org/wiki/Single-responsibility_principle

To manage these containers we use docker-compose. The Dockerfiles that contain the build instructions are still called by the docker-compose command. I will expand more on this when I discuss how to run the container. First let's install docker-compose:

To install docker-compose we will use curl. If for some reason you do not have curl you can install it using:

```console
apt-get install curl
```

Now we will use curl to download docker-compose into /usr/local/bin/

```console 
sudo curl -L "https://github.com/docker/compose/releases/download/1.29.1/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
```
For security reasons files downloaded from the web are not executable. To make the docker-compose binary executable run 

```console
sudo chmod +x /usr/local/bin/docker-compose
```

Finally test that the installation was successful:

```console
docker-compose --version
```

### Run the Jupyter container

As things stand we are not using cuda to allow for GPU acceleration. This will soon change as we are actively working to update the container to include cuda. Hence for the following instructions assume that cuda is not present. 

Now that all of the tools required to run the Jupyter container have been installed we can setup a local project repository using `git clone`. I will assume that you have setup an ssh key for your machine. 

Navigate to a directory where you wish to place project files. You may wish to make a folder in your home directory for this purpose. For example if you wanted to house all of your code in a folder called projects you would run `mkdir ~/projects` then navigate to this folder using `cd ~/projects`. 

From here you can pull down the repository using 

```console
git clone git@github.com:sdwalker62/log-analyzer.git
```
This will create a new folder in your current directory called log-analyzer. Navigate to this folder using `cd log-analyzer`. In this folder is the docker-compose.yaml which contains the instructions for docker-compose as well as the project's sub-directories described below. From here you can run 

```console
sudo docker-compose up jupyter 
```
or 

```console
sudo docker-compose up --build jupyter
```

if you have made changes to either the docker-compose file or the Dockerfile in /jupyter/

## Jupyter 

This is the main directory for experimentation. It houses all the code necessary to build the jupyter docker container. The notebooks, results, data, and doc directories are mapped to this container through docker-compose for serializing/deserializing objects, loading trained models, and modifying LaTeX reports. 

## Notebooks 

This is source folder for all Jupyter notebooks. As Jupyter is the primary tool for experimentation this folder houses all experimental code. Updates and modifications to the pre-processing and training folders more than likely originate from this directory. Currently there is only the longruntransformer.ipynb notebook in this directory which holds the experimental transformer code. I will soon add a playground.ipynb notebook for throwaway code. 

## Docs 

## Pre-processing 

## Training
