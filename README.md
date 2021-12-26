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
   - 2.1 [Installation](#2.1-installation)

# 1. Image Hierarchy

The current heirarchy of images can be found below. A description of each image can be found in the docs folder. The docs folder contains auto-generated documentation that is updated on every build to insure always up-to-date information about each image. Feel free to use any of the images found in this repository as a base for your own custom images. I will keep these images updated regularly from the upstream repositories.

<details>
<summary> click to reveal image hierachy diagram </summary>

![Image Hierarchy](assets/image_relations.svg "Image Hierarchy")

</details>

please

# 2. Usage Instructions

In this section we will discuss how we intend these images to be ran as well as how to install the required components on the host machine.

The easiest way to use any of the images is to modify the included `docker-compose.yml` and run `docker-compose up`{:.bash} to start Jupyterlab.  
