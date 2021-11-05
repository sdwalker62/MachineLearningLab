# Inspiration for this Dockerfile comes from the gpu-jupyter repository by cschranz. 
# You can find his repository at https://github.com/iot-salzburg/gpu-jupyter 

FROM nvidia/cuda:11.4.2-cudnn8-runtime-ubuntu20.04
LABEL authors="Dalton Walker <samuel.dalton.walker@gmail.com>, Andrew Romans <>, Michael Fox <>"
LABEL team=SIGIL
LABEL date=2021-11-04



# [0] Base Layer ............. #
RUN chmod 1777 /tmp /var/tmp

ARG NB_USER="jovyan"
ARG NB_UID="1000"
ARG NB_GID="100"

# Fix DL4006
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER root

# Install all OS dependencies for notebook server that starts but lacks all
# features (e.g., download as all possible file formats)
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get -q update \
 && apt-get install -yq --no-install-recommends \
    wget \
    ca-certificates \
    sudo \
    locales \
    fonts-liberation \
    run-one \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen

# Enable prompt color in the skeleton .bashrc before creating the default NB_USER
# hadolint ignore=SC2016
RUN sed -i 's/^#force_color_prompt=yes/force_color_prompt=yes/' /etc/skel/.bashrc && \
   # Add call to conda init script see https://stackoverflow.com/a/58081608/4413446
   echo 'eval "$(command conda shell.bash hook 2> /dev/null)"' >> /etc/skel/.bashrc 

# Create NB_USER with name jovyan user with UID=1000 and in the 'users' group
# and make sure these dirs are writable by the `users` group.
RUN echo "auth requisite pam_deny.so" >> /etc/pam.d/su && \
    sed -i.bak -e 's/^%admin/#%admin/' /etc/sudoers && \
    sed -i.bak -e 's/^%sudo/#%sudo/' /etc/sudoers && \
    useradd -m -s /bin/bash -N -u ${NB_UID} ${NB_USER} && \
    chmod g+w /etc/passwd

WORKDIR /tmp

# [0.0] Install miniconda #
# Configure environment
ENV CONDA_DIR=/opt/conda \
    SHELL=/bin/bash \
    NB_USER=$NB_USER \
    NB_UID=$NB_UID \
    NB_GID=$NB_GID \
    LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8
ENV PATH=$CONDA_DIR/bin:$PATH \
    HOME=/home/$NB_USER \
    CONDA_VERSION="4.10.3" 

ARG SHA256_HASH="1ea2f885b4dbc3098662845560bc64271eb17085387a70c2ba3f29fff6f8d52f"
ARG ARCH="x86_64"
ARG MINICONDA_INSTALLER="Miniconda3-py39_4.10.3-Linux-x86_64.sh"

# Install and update miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh && \
    # verify the installer's checksum
    echo "${SHA256_HASH} *${MINICONDA_INSTALLER}" | sha256sum --check && \    
    # create conda's directory
    mkdir -p $CONDA_DIR && \
    # change ownership of the directory to the notebooks user and group
    chown ${NB_USER}:${NB_GID} ${CONDA_DIR} && \
    # execute the installer
    ${SHELL} "${MINICONDA_INSTALLER}" -f -b -p ${CONDA_DIR} && \
    # cleanup the installer's script
    rm "${MINICONDA_INSTALLER}" && \
    # pin the current version of conda so it will not update
    echo "conda ${CONDA_VERSION}" >> $CONDA_DIR/conda-meta/pinned && \
    # disable auto-updating
    conda config --system --set auto_update_conda false && \
    # toggle display information
    conda config --system --set show_channel_urls true && \
    # install correct Python version
    if [ ! $PYTHON_VERSION = 'default' ]; then conda install --yes python=$PYTHON_VERSION; fi && \
    # pin this version of Python so it will not update and break packages
    conda list python | grep '^python ' | tr -s ' ' | cut -d '.' -f 1,2 | sed 's/$/.*/' >> $CONDA_DIR/conda-meta/pinned && \
    # install correct conda version, pip, and tini
    conda install --quiet --yes \
    "conda=${CONDA_VERSION}" \
    'pip' \
    'tini=0.19.0' && \
    # force updatge all conda packages
    conda update --all --quiet --yes && \
    # pin tini (https://github.com/krallin/tini)
    conda list tini | grep tini | tr -s ' ' | cut -d ' ' -f 1,2 >> $CONDA_DIR/conda-meta/pinned && \
    # remove unused packages
    conda clean --all -f -y && \
    # remove the Yarn cache (for more information on Yarn: https://classic.yarnpkg.com/en/docs)
    rm -rf /home/$NB_USER/.cache/yarn

ENTRYPOINT ["tini", "-g", "--"]


# [0.1] Install Jupyter packages
RUN conda install --quiet --yes \
    'notebook' \
    'jupyterhub' \
    'jupyterlab' && \
    conda clean --all -f -y && \
    npm cache clean --force && \
    jupyter notebook --generate-config && \
    jupyter lab clean && \
    rm -rf /home/$NB_USER/.cache/yarn

EXPOSE 8888

CMD ["start-notebook.sh"]












ARG FONT_DIR=/usr/local/share/fonts

USER root

RUN jupyter labextension install jupyterlab-tailwind-theme

RUN mkdir $FONT_DIR/FiraCode/ && \
    chown $NB_UID:$NB_GID $FONT_DIR/FiraCode

COPY assets/FiraCode-Retina.ttf $FONT_DIR/FiraCode
COPY config/ /home/$NB_USER/.jupyter/

USER $NB_UID