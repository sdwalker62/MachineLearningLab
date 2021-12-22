  __  __         _    _            _                      _             _         _    
 |  \/  |__ _ __| |_ (_)_ _  ___  | |   ___ __ _ _ _ _ _ (_)_ _  __ _  | |   __ _| |__ 
 | |\/| / _` / _| ' \| | ' \/ -_) | |__/ -_) _` | '_| ' \| | ' \/ _` | | |__/ _` | '_ \
 |_|  |_\__,_\__|_||_|_|_||_\___| |____\___\__,_|_| |_||_|_|_||_\__, | |____\__,_|_.__/
                                                                |___/                 

This repository is based heavily on the docker-stacks repository (https://github.com/jupyter/docker-stacks). We primarily use containers with cuda baked into the image and hence don't use the images on the jupyter docker hub. We found it easier to base our images on the nvidia/cuda images instead of basing our images on the docker-stack images and installing cuda on top. If you aren't using cuda then it would be advisable to checkout the original docker-stacks at the above link. If you want another implementation cuda with jupyter checkout the excellent images found here (https://github.com/iot-salzburg/gpu-jupyter/). 

We want to thank the Jupyter team for making and sustaining a wonderful set of tools for data scientist and hope anyone reading this finds this tool stack helpful. 

---



