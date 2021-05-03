# set base image (host OS)
FROM jupyter/tensorflow-notebook 

USER root

#RUN yes | apt-get update 
#RUN yes | apt-get upgrade

# copy the dependencies file to the working directory
COPY requirements.txt requirements.txt

# install dependencies
RUN pip install -r requirements.txt

#RUN apt-get update && apt-get install -y --no-install-recommends \
#    cuda-cudart-11-0=11.0.221-1 \
#    cuda-compat-11-0 && \
#    ln -s cuda-11.0 /usr/local/cuda && \
#    rm -rf /var/lib/apt/lists/* # buildkit

#RUN apt-get update && apt-get install -y --no-install-recommends build-essential \ 
#	cuda-command-line-tools-${CUDA/./-} \
#        libcublas-${CUDA/./-} \
#        cuda-nvrtc-${CUDA/./-} \
#        libcufft-${CUDA/./-} \ 
#        libcurand-${CUDA/./-} \
#        libcusolver-${CUDA/./-} \
#        libcusparse-${CUDA/./-}

RUN apt-get install libcusolver-11.0

RUN jupyter labextension install \
    @jupyter-widgets/jupyterlab-manager \
    jupyterlab-system-monitor \
    @jupyterlab/debugger
   
COPY config/ /home/jovyan/.jupyter/

# install FiraCode font
COPY FiraCode-Retina.ttf /home/jovyan/.fonts
RUN fc-cache -f -v


# command to run on container start
CMD ["jupyter", "lab","--ip=0.0.0.0", "--allow-root"]
