```
 __  __            _     _              _                          _               _          _
|  \/  | __ _  ___| |__ (_)_ __   ___  | |    ___  __ _ _ __ _ __ (_)_ __   __ _  | |    __ _| |__
| |\/| |/ _` |/ __| '_ \| | '_ \ / _ \ | |   / _ \/ _` | '__| '_ \| | '_ \ / _` | | |   / _` | '_ \
| |  | | (_| | (__| | | | | | | |  __/ | |__|  __/ (_| | |  | | | | | | | | (_| | | |__| (_| | |_) |
|_|  |_|\__,_|\___|_| |_|_|_| |_|\___| |_____\___|\__,_|_|  |_| |_|_|_| |_|\__, | |_____\__,_|_.__/
                                                                           |___/

```

<table border="1">
    <tr>
        <td>
            <a href="https://hub.docker.com/r/samuel62/machine_learning_lab"> 
                <img src="https://www.docker.com/sites/default/files/d8/2019-07/horizontal-logo-monochromatic-white.png" alt="Dockerhub" height="40"/> 
            </a>
        </td>
        <td>
            <a href="docs">
            <img src="assets/docs.png" alt="Documentation" height="40"/>
            </a>
        </td>
        <td>
            <a href="https://gitlab.com/samuel.dalton.walker/MachineLearningLab">
            <img src="https://about.gitlab.com/images/press/logo/svg/gitlab-logo-gray-rgb.svg" alt="Gitlab" height="40"/>
            </a>
        </td>
    </tr>
</table>

Welcome to the Machine Learning Lab repository! This repository is based heavily on the fantastic work being done at the [jupyter/docker-stacks](https://github.com/jupyter/docker-stacks) repository, if you haven't checked it out please do and support the work being done there. In this reposity we include both cuda-enabled versions of a select number of images from the docker-stacks repository as well as some custom images not found there. If you would like to add images to this repository we welcome that - just open a pull request! We plan to include a contribute.md soon that will outline how contributions should be made!

# 0. Table of Contents

1. [Image Hierarchy](#1-image-hierarchy)
2. [Usage Instructions](#2-usage-instructions)
    - 2.1 [Installation](##2.1-installation)

# 1. Image Hierarchy

The current heirarchy of images can be found below. A description of each image can be found in the docs folder. The docs folder contains auto-generated documentation that is updated on every build to insure always up-to-date information about each image. Feel free to use any of the images found in this repository as a base for your own custom images. I will keep these images updated regularly from the upstream repositories.

<details>
<summary> click to reveal image hierachy diagram </summary>

![Image Hierarchy](assets/image_relations.svg "Image Hierarchy")

</details>

please

# 2. Usage Instructions

In this section we will discuss how we intend these images to be ran as well as how to install the required components on the host machine.

The easiest way to use any of the images is to modify the included docker-compose.yml and run `docker-compose up` to start Jupyterlab. For instance an example docker-compose.yml would be: 

<details>
<summary> click to reveal docker-compose.yml </summary>

```yaml
version: "3.8"

services:
  machine_learning_lab:
    image: IMAGE_NAME
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ports:
      - 8888:8888
    environment:
      - JUPYTER_ENABLE_LAB=1
    volumes:
      - data:data:rw
      - results:results:rw
```

In this example docker-compose.yml you would replace *IMAGE_NAME* with the name of whatever image you wish to run, for instance: `samuel62/machine_learning_lab:datascience_cuda_11.3.1`

Since most users of the images found in this repository are data scientist we recommend binding a volume from your host machine to the container in the volumes section. Typically, you would include a *data* and a *results* folder in this section as well as any other directories you want the container to be able to read/write from on your host machine. 

If you are not using a machine with a dedicated Nvidia GPU then comment/remove the deploy section as it will cause an error. 

</details>

\
It is not necessary to use `docker-compose` to run the containers. Instead you could run them with the `docker run` command, e.g.:

```docker
docker run -p 8888:8888 -e JUPYTER_ENABLE_LAB=1 -v data:data:ro -v results:results:rw samuel62/machine_learning_lab:datascience_cuda_11.3.1
```

or 

```docker
docker run \
    -p 8888:8888 \
    -e JUPYTER_ENABLE_LAB=1 \
    -v data:data:ro \
    -v results:results:rw \
    samuel62/machine_learning_lab:datascience_cuda_11.3.1
```

We recommend using the docker-compose method as it allows for more control without atrocious syntax. 

## 2.1 Installation 

### Linux Instructions
<details>
<summary> reveal Linux instructions </summary>

There are two requirements on Linux and one optional command. The first is docker which we will show how to install for Debian (which includes Ubuntu) based distributions. 

---

```
  ___          _           
 |   \ ___  __| |_____ _ _ 
 | |) / _ \/ _| / / -_) '_|
 |___/\___/\__|_\_\___|_|  
                           
```

If you don't know what docker is please read this for more information: [Docker overview](https://docs.docker.com/get-started/overview/)

There are two ways to install Docker on Linux, one is from a convenience script and the other is manually. We will demonstrate both. 

\
***Convenience Scipt***
\
The convenience script will attempt to install the most recent version of Docker on your machine. All that you need to do is curl the script and execute it using root privileges.

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
DRY_RUN=1 sh ./get-docker.sh
```
\
***Manual Installation***
\
The instructions below come from the official Docker installation instructions for Ubuntu found here: [official instructions](https://docs.docker.com/engine/install/ubuntu/)

\
***Check for old installations***
\
First we need to check for any current docker installations and remove them. If you are installing on a new Linux install then skip this section, otherwise it is recommended to run the following command: 

```bash
sudo apt-get remove docker docker-engine docker.io containerd runc
```

If `apt-get` reports that none of those packages are installed that is OK and you can continue.

\
***Pre-requisites***
\
Update the apt package index: 

```bash
sudo apt-get update
```

Install the following packages to allow `apt` to use https:

```bash
sudo apt-get install \
    ca-certificates \
    curl \
    gnupg \
    lsb-release
```

To install docker using apt-get we will need to add the docker gpg-key to sources.list. For security we can't install any packages that haven't been signed. 

```bash
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

After adding the gpg-key we can now install the docker-engine. First we will update the `apt` package list again.
```bash
sudo apt-get update
```

Now we can install the most current version of the docker-engine:

```bash
sudo apt-get install docker-ce docker-ce-cli containerd.io
```

\
***Post installation steps***
\
This section is optional but recommended. Once you have installed Docker on your machine you will be required to run any docker commands as the root using `sudo`. This is inconvenient among other things and hence adding the current user to the docker group is advised. Follow the steps in this section to add yourself to the docker group. 

First we must create the docker group:

```bash
sudo groupadd docker
```

Now add yourself to the group:

```bash
sudo usermod -aG docker $USER
```

Finally run the follwoing command to activate the changes to groups:

```bash
newgrp docker
```

With all of the above complete you are finished installing Docker! You can test your installation by running `docker info` or `docker run hello-world`.

---

```
  _  ___   _____ ___ ___   _      ___ _____ _  __
 | \| \ \ / /_ _|   \_ _| /_\    / __|_   _| |/ /
 | .` |\ V / | || |) | | / _ \  | (__  | | | ' < 
 |_|\_| \_/ |___|___/___/_/ \_\  \___| |_| |_|\_\
                                                 
```
**NOTE**: This section is only relevant for those intending to run the images with GPU support. If your host machine does not have a dedicated NVIDIA GPU then skip this section and remember to run the images without the `--gpus all` flag or the `deploy` section if using `docker-compose`

The following information is summarized from the [offical guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html), please refer to the official guide for clarification on any of the below steps.

There are a few pre-installation steps we must take before installing the container toolkit. 

* Verify the system has a CUDA-capable GPU.
* Verify the system is running a supported version of Linux.
* Verify the system has gcc installed.
* Verify the system has the correct kernel headers and development packages installed.
* Download the NVIDIA CUDA Toolkit.
* Handle conflicting installation methods.

To verify that the system has a CUDA-capable GPU run 
```bash
lspci | grep -i nvidia
```
Make sure a CUDA-capable GPU is listed in the output. If not stop. If you believe there is an error and your GPU is not found then update the PCI hardware database using `update-pciids` and try the above command again. 

To verify that the system is running a support distribution run

```bash
uname -m && cat /etc/*release
```
and check the below table to make sure your distribution is supported. If it isn't then either skip this section or install one of the supported distributions.

---

<details>
<summary> click to reveal supported distributions </summary>
<table summary="" class="table" frame="border" rules="all" cellspacing="0" cellpadding="4" border="1">
    <caption><span class="tablecap">Table 1. Native Linux Distribution Support in CUDA <span class="keyword">11.5</span></span></caption>
    <thead class="thead" align="left">
        <tr class="row">
        <th class="entry" id="d117e166" rowspan="1" colspan="1" width="NaN%" valign="top">Distribution</th>
        <th class="entry" id="d117e169" rowspan="1" colspan="1" width="NaN%" valign="top">Kernel<sup class="ph sup">1</sup></th>
        <th class="entry" id="d117e174" rowspan="1" colspan="1" width="NaN%" valign="top">Default GCC</th>
        <th class="entry" id="d117e177" rowspan="1" colspan="1" width="NaN%" valign="top">GLIBC</th>
        <th class="entry" id="d117e180" rowspan="1" colspan="1" width="NaN%" valign="top">GCC<sup class="ph sup">2,3</sup></th>
        <th class="entry" id="d117e186" rowspan="1" colspan="1" width="NaN%" valign="top">ICC<sup class="ph sup">3</sup></th>
        <th class="entry" id="d117e191" rowspan="1" colspan="1" width="NaN%" valign="top">NVHPC<sup class="ph sup">3</sup></th>
        <th class="entry" id="d117e196" rowspan="1" colspan="1" width="NaN%" valign="top">XLC<sup class="ph sup">3</sup></th>
        <th class="entry" id="d117e201" rowspan="1" colspan="1" width="NaN%" valign="top">CLANG</th>
        <th class="entry" id="d117e204" rowspan="1" colspan="1" width="NaN%" valign="top">Arm C/C++</th>
        </tr>
    </thead>
    <tbody class="tbody">
        <tr class="row gray">
        <td class="entry" colspan="10" headers="d117e166 d117e169 d117e174 d117e177 d117e180 d117e186 d117e191 d117e196 d117e201 d117e204" rowspan="1" valign="top" align="center">x86_64</td>
        </tr>
        <tr class="row">
        <td class="entry" headers="d117e166" rowspan="1" colspan="1" width="NaN%" valign="top">RHEL 8.y (y &lt;= 4)</td>
        <td class="entry" headers="d117e169" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">4.18.0-305</td>
        <td class="entry" headers="d117e174" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">8.4.1</td>
        <td class="entry" headers="d117e177" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">2.28</td>
        <td class="entry" rowspan="10" headers="d117e180" colspan="1" width="NaN%" valign="middle" align="center">11</td>
        <td class="entry" rowspan="10" headers="d117e186" colspan="1" width="NaN%" valign="middle" align="center">2021</td>
        <td class="entry" rowspan="10" headers="d117e191" colspan="1" width="NaN%" valign="middle" align="center">21.7</td>
        <td class="entry" rowspan="10" headers="d117e196" colspan="1" width="NaN%" valign="middle" align="center">NO</td>
        <td class="entry" rowspan="10" headers="d117e201" colspan="1" width="NaN%" valign="middle" align="center">12</td>
        <td class="entry" rowspan="10" headers="d117e204" colspan="1" width="NaN%" valign="middle" align="center">NO</td>
        </tr>
        <tr class="row">
        <td class="entry" headers="d117e166" rowspan="1" colspan="1" width="NaN%" valign="top">CentOS 8.y (y &lt;= 4)</td>
        <td class="entry" headers="d117e169" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">4.18.0-305</td>
        <td class="entry" headers="d117e174" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">8.4.1</td>
        <td class="entry" headers="d117e177" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">2.28</td>
        </tr>
        <tr class="row">
        <td class="entry" headers="d117e166" rowspan="1" colspan="1" width="NaN%" valign="top">RHEL 7.y (y &lt;= 9)</td>
        <td class="entry" headers="d117e169" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">3.10.0-1160</td>
        <td class="entry" headers="d117e174" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">6.x</td>
        <td class="entry" headers="d117e177" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">2.17</td>
        </tr>
        <tr class="row">
        <td class="entry" headers="d117e166" rowspan="1" colspan="1" width="NaN%" valign="top">CentOS 7.y (y &lt;= 9)</td>
        <td class="entry" headers="d117e169" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">3.10.0-1160</td>
        <td class="entry" headers="d117e174" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">6.x</td>
        <td class="entry" headers="d117e177" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">2.17</td>
        </tr>
        <tr class="row">
        <td class="entry" headers="d117e166" rowspan="1" colspan="1" width="NaN%" valign="top">OpenSUSE Leap 15.y (y &lt;= 3)</td>
        <td class="entry" headers="d117e169" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">5.3.18-57</td>
        <td class="entry" headers="d117e174" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">7.5.0</td>
        <td class="entry" headers="d117e177" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">2.31</td>
        </tr>
        <tr class="row">
        <td class="entry" headers="d117e166" rowspan="1" colspan="1" width="NaN%" valign="top">SUSE SLES 15.y (y &lt;= 3)</td>
        <td class="entry" headers="d117e169" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">5.3.18-57</td>
        <td class="entry" headers="d117e174" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">7.5.0</td>
        <td class="entry" headers="d117e177" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">2.31</td>
        </tr>
        <tr class="row">
        <td class="entry" headers="d117e166" rowspan="1" colspan="1" width="NaN%" valign="top">Ubuntu 20.04.3</td>
        <td class="entry" headers="d117e169" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">5.11.0-27</td>
        <td class="entry" headers="d117e174" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">9.3.0</td>
        <td class="entry" headers="d117e177" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">2.31</td>
        </tr>
        <tr class="row">
        <td class="entry" headers="d117e166" rowspan="1" colspan="1" width="NaN%" valign="top">Ubuntu 18.04.z (z &lt;= 6)</td>
        <td class="entry" headers="d117e169" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">5.4.0-89</td>
        <td class="entry" headers="d117e174" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">7.5.0</td>
        <td class="entry" headers="d117e177" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">2.27</td>
        </tr>
        <tr class="row">
        <td class="entry" headers="d117e166" rowspan="1" colspan="1" width="NaN%" valign="top">Debian 11.1</td>
        <td class="entry" headers="d117e169" rowspan="1" colspan="1" width="NaN%" valign="top">5.10.0-9</td>
        <td class="entry" headers="d117e174" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">10.2.1</td>
        <td class="entry" headers="d117e177" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">2.31</td>
        </tr>
        <tr class="row">
        <td class="entry" headers="d117e166" rowspan="1" colspan="1" width="NaN%" valign="top">Fedora 34</td>
        <td class="entry" headers="d117e169" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">5.11</td>
        <td class="entry" headers="d117e174" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">11</td>
        <td class="entry" headers="d117e177" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">2.33</td>
        </tr>
        <tr class="row gray">
        <td class="entry" colspan="10" headers="d117e166 d117e169 d117e174 d117e177 d117e180 d117e186 d117e191 d117e196 d117e201 d117e204" rowspan="1" valign="middle" align="center">Arm64 sbsa</td>
        </tr>
        <tr class="row">
        <td class="entry" headers="d117e166" rowspan="1" colspan="1" width="NaN%" valign="top">RHEL 8.4</td>
        <td class="entry" headers="d117e169" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">4.18.0-305</td>
        <td class="entry" headers="d117e174" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">8.4.1</td>
        <td class="entry" headers="d117e177" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">2.28</td>
        <td class="entry" rowspan="3" headers="d117e180" colspan="1" width="NaN%" valign="middle" align="center">11</td>
        <td class="entry" rowspan="3" headers="d117e186" colspan="1" width="NaN%" valign="middle" align="center">NO</td>
        <td class="entry" rowspan="3" headers="d117e191" colspan="1" width="NaN%" valign="middle" align="center"> 21.7</td>
        <td class="entry" rowspan="3" headers="d117e196" colspan="1" width="NaN%" valign="middle" align="center">NO</td>
        <td class="entry" rowspan="3" headers="d117e201" colspan="1" width="NaN%" valign="middle" align="center">12</td>
        <td class="entry" rowspan="3" headers="d117e204" colspan="1" width="NaN%" valign="middle" align="center">21.0</td>
        </tr>
        <tr class="row">
        <td class="entry" headers="d117e166" rowspan="1" colspan="1" width="NaN%" valign="top">SUSE SLES 15.y (y &lt;= 3)</td>
        <td class="entry" headers="d117e169" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">5.3.18-57</td>
        <td class="entry" headers="d117e174" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">7.5.0</td>
        <td class="entry" headers="d117e177" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">2.31</td>
        </tr>
        <tr class="row">
        <td class="entry" headers="d117e166" rowspan="1" colspan="1" width="NaN%" valign="top">Ubuntu 20.04.3</td>
        <td class="entry" headers="d117e169" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">5.4.0-86</td>
        <td class="entry" headers="d117e174" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">9.3.0</td>
        <td class="entry" headers="d117e177" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">2.31</td>
        </tr>
        <tr class="row gray">
        <td class="entry" colspan="10" headers="d117e166 d117e169 d117e174 d117e177 d117e180 d117e186 d117e191 d117e196 d117e201 d117e204" rowspan="1" valign="middle" align="center">Arm64 Jetson</td>
        </tr>
        <tr class="row">
        <td class="entry" headers="d117e166" rowspan="1" colspan="1" width="NaN%" valign="top">Ubuntu 18.04.z (z &lt;= 6)</td>
        <td class="entry" headers="d117e169" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">4.9.253</td>
        <td class="entry" headers="d117e174" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">7.5.0</td>
        <td class="entry" headers="d117e177" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">2.27</td>
        <td class="entry" headers="d117e180" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">10.2</td>
        <td class="entry" headers="d117e186" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">NO</td>
        <td class="entry" headers="d117e191" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">21.7</td>
        <td class="entry" headers="d117e196" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">16.1.x</td>
        <td class="entry" headers="d117e201" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">12</td>
        <td class="entry" headers="d117e204" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">NO</td>
        </tr>
        <tr class="row gray">
        <td class="entry" colspan="10" headers="d117e166 d117e169 d117e174 d117e177 d117e180 d117e186 d117e191 d117e196 d117e201 d117e204" rowspan="1" valign="middle" align="center">POWER 9</td>
        </tr>
        <tr class="row">
        <td class="entry" headers="d117e166" rowspan="1" colspan="1" width="NaN%" valign="top">RHEL 8.y (y &lt;= 4)</td>
        <td class="entry" headers="d117e169" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">4.18.0-240</td>
        <td class="entry" headers="d117e174" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">8.3.1</td>
        <td class="entry" headers="d117e177" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">2.28</td>
        <td class="entry" headers="d117e180" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">11</td>
        <td class="entry" headers="d117e186" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">NO</td>
        <td class="entry" headers="d117e191" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">21.7</td>
        <td class="entry" headers="d117e196" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">16.1.x</td>
        <td class="entry" headers="d117e201" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">12</td>
        <td class="entry" headers="d117e204" rowspan="1" colspan="1" width="NaN%" valign="middle" align="center">NO</td>
        </tr>
    </tbody>
</table>
</details>   

---

To verify that your system has `gcc` installed run 
```bash
gcc --version
```
If an error message is returned then please install `gcc` and its accompanying toolchain. On Ubuntu this can be achieved by running

```bash
sudo apt install build-essential
```
To check what verion of the kernel headers you have run 

```bash
uname -r
```

Cross reference the returned value with the one listed for your distribution in the above table. If you need to update your kernel headers follow the below instructions:

If you are running Ubuntu and need to install the newest kernel headers run 

```bash
sudo apt-get install linux-headers-$(uname -r)
```

for all other distributions refer to official guide for the appropriate commands. 

*** Install CTK ***

Now we can install the NVIDIA CUDA Toolkit. The toolkit come with everything needed to run CUDA applications including drivers, header files, etc.. We will need this to run cuda-enabled images. 

Follow the instructions here for your platform: https://developer.nvidia.com/cuda-downloads

---

```
  ___          _                ___                             
 |   \ ___  __| |_____ _ _ ___ / __|___ _ __  _ __  ___ ___ ___ 
 | |) / _ \/ _| / / -_) '_|___| (__/ _ \ '  \| '_ \/ _ (_-</ -_)
 |___/\___/\__|_\_\___|_|      \___\___/_|_|_| .__/\___/__/\___|
                                             |_|                
```

This section is optional as you can run any of the images without using docker-compose. We recommend installing docker-compose as it makes configuring the container much easier and allows for complex docker setups.

We can curl the binary directly into the bin directory:

```bash
sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
 ```

 Now apply executable permissions to the binary:

```bash
sudo chmod +x /usr/local/bin/docker-compose
```

---

This completes the pre-requisites to run the images in this repository. Next we will talk about how to run the images.

---