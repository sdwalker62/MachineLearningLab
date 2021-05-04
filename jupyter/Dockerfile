# Set base image (host OS)
FROM ubuntu:latest

# miniconda does not work with the sh shell
# --login ensures ~/.profile and ~/.bashrc 
# are sourced properly 
SHELL [ "/bin/bash", "--login", "-c" ]


# -- Install wget for conda -- #
RUN  apt-get update \
  && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/* 

# -- User creation -- #
ARG username
ARG uid
ARG gid
ENV USER $username
ENV UID $uid
ENV GID $gid
ENV HOME /home/$USER
RUN adduser --disabled-password \
    --gecos "Non-root user" \
    --uid $UID \
    --gid $GID \
    --home $HOME \
    $USER

# -- Copy config files -- # 
COPY environment.yml requirements.txt /tmp/
RUN chown $UID:$GID /tmp/environment.yml /tmp/requirements.txt

COPY jupyterextensions.sh /usr/local/bin/jupyterextensions.sh
RUN chown $UID:$GID /usr/local/bin/jupyterextensions.sh && \
    chmod u+x /usr/local/bin/jupyterextensions.sh

COPY entrypoint.sh /usr/local/bin/
RUN chown $UID:$GID /usr/local/bin/entrypoint.sh && \
    chmod u+x /usr/local/bin/entrypoint.sh

COPY config/ /home/$USER/.jupyter/

RUN chown -R $USER:$GID /home/${USER}/.jupyter/

# -- Directory creation --

# Env dir
ENV ENV_PREFIX $PWD/env
RUN mkdir $ENV_PREFIX && \
    chown $UID:$GID $ENV_PREFIX

# Font dir
RUN mkdir /usr/local/share/fonts
ENV FONT_DIR /usr/local/share/fonts
RUN mkdir $FONT_DIR/FiraCode/ && \
    chown $UID:$GID $FONT_DIR/FiraCode

# Project dir
ENV PROJECT_DIR $HOME/app
RUN mkdir $PROJECT_DIR && \
    chown $UID:$GID $PROJECT_DIR && \
    chown $UID:$GID $HOME

RUN chown -R $USER:$GID /home/$USER/app/


# -- Install miniconda -- #
USER $USER
# install miniconda
ENV MINICONDA_VERSION py39_4.9.2
ENV CONDA_DIR $HOME/miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-$MINICONDA_VERSION-Linux-x86_64.sh -O ~/miniconda.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh
# make non-activate conda commands available
ENV PATH=$CONDA_DIR/bin:$PATH
# make conda activate command available from /bin/bash --login shells
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile
# make conda activate command available from /bin/bash --interative shells
RUN conda init bash

WORKDIR $PROJECT_DIR


# -- Conda environment creation -- #
RUN conda update --name base --channel defaults conda && \
    conda env create --prefix $ENV_PREFIX --file /tmp/environment.yml --force && \
    conda clean --all --yes


# -- Jupyterlab extensions installation -- #
RUN conda activate $ENV_PREFIX && \
    /usr/local/bin/jupyterextensions.sh && \
    conda deactivate


# Install FiraCode font (the best font)
COPY FiraCode-Retina.ttf $FONT_DIR/FiraCode
   
# Copy config files 


# Default command will launch JupyterLab server for development
ENTRYPOINT [ "/usr/local/bin/entrypoint.sh" ]
CMD ["jupyter", "lab","--ip=0.0.0.0", "--allow-root"]
#CMD [ "jupyter", "lab", "--no-browser", "--ip", "0.0.0.0", "--allow-root" ]