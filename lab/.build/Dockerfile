# Inspiration for this Dockerfile comes from the gpu-jupyter repository by cschranz. 
# You can find his repository at https://github.com/iot-salzburg/gpu-jupyter 

FROM nvidia/cuda:11.4.2-cudnn8-runtime-ubuntu20.04
LABEL authors="Dalton Walker <samuel.dalton.walker@gmail.com>, Andrew Romans <andrew.romans@hotmail.com>, Michael Fox <mjfox314@gmail.com>"
LABEL team=SIGIL
LABEL date=2021-11-15

# TOC: 
# SN  Item                      Line Number 
# =========================================
# [0] Base Layer .............. 22
# [1] Install Anaconda ........ 98
# [2] Install Jupyter ......... 132
# [3] Install Conda Packages .. 178
# [4] Install pip Packages .... 214
# [5] Install Jupyterlab Ext .. 222
# [6] Install Dynamic Content . 238
# =========================================


# [0] Base Layer ............. #
RUN chmod 1777 /tmp /var/tmp

ARG NB_USER="jovyan"
ARG NB_UID="1000"
ARG NB_GID="100"

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
    CONDA_VERSION="${conda_version}" \
    MINIFORGE_VERSION="${miniforge_version}"

# Fix DL4006
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER root

# Copy a script that we will use to correct permissions after running certain commands
COPY scripts/fix-permissions /usr/local/bin/fix-permissions
RUN chmod a+rx /usr/local/bin/fix-permissions

# Add package lists directory which we can remove later
ARG PKG_DIR="/pkg_lists"
RUN mkdir $PKG_DIR

# Copy the static elements only to avoid cache problems
COPY /static-elements/* \
     /scripts/* \
     ${PKG_DIR}

RUN apt-get -q update && apt-get install -yq locales && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen

# Instructions located at https://docs.anaconda.com/anaconda/install/linux/
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get -q update \
 && xargs -a ${PKG_DIR}/static-apt-packages.txt \
 apt-get install -yq --no-install-recommends \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

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
    useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
    mkdir -p $CONDA_DIR && \
    chown $NB_USER:$NB_GID $CONDA_DIR && \
    chmod g+w /etc/passwd && \
    fix-permissions $HOME && \
    fix-permissions $CONDA_DIR

USER $NB_UID
ARG PYTHON_VERSION=default

# Setup work directory for backward-compatibility
RUN mkdir "/home/$NB_USER/work" && \
    fix-permissions "/home/$NB_USER"


# [1] Install Anaconda ............. #

# Install conda as jovyan and check the sha256 sum provided on the download site
WORKDIR /tmp

#https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
ARG ANACONDA_VERSION="2021.05"
ARG SYSTEM_ARCH="x86_64"
ARG ANACONDA_INSTALLER="Anaconda3-${ANACONDA_VERSION}-Linux-${SYSTEM_ARCH}.sh"
ARG SHA256_HASH="2751ab3d678ff0277ae80f9e8a74f218cfc70fe9a9cdc7bb1c137d7e47e33d53"

# Prerequisites installation: conda, mamba, pip, tini
RUN wget --quiet "https://repo.anaconda.com/archive/${ANACONDA_INSTALLER}" && \
    echo "${SHA256_HASH} *${ANACONDA_INSTALLER}" | sha256sum --check && \
    /bin/bash "${ANACONDA_INSTALLER}" -f -b -p $CONDA_DIR && \
    rm "${ANACONDA_INSTALLER}" && \
    # Conda configuration see https://conda.io/projects/conda/en/latest/configuration.html
    echo "conda ${CONDA_VERSION}" >> $CONDA_DIR/conda-meta/pinned && \
    conda config --system --set auto_update_conda false && \
    conda config --system --set show_channel_urls true && \
    if [ ! $PYTHON_VERSION = 'default' ]; then conda install --yes python=$PYTHON_VERSION; fi && \
    conda list python | grep '^python ' | tr -s ' ' | cut -d '.' -f 1,2 | sed 's/$/.*/' >> $CONDA_DIR/conda-meta/pinned && \
    conda install --quiet --yes \
    "conda=${CONDA_VERSION}" \
    'pip' \
    'tini=0.18.0' && \
    conda update --all --quiet --yes && \
    conda list tini | grep tini | tr -s ' ' | cut -d ' ' -f 1,2 >> $CONDA_DIR/conda-meta/pinned && \
    conda clean --all -f -y && \
    rm -rf /home/$NB_USER/.cache/yarn && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER


# [2] Install Jupyter from Conda ............. #

# Install Jupyter Lab 
# Generate a notebook server config
# Cleanup temporary files
# Correct permissions
# Do all this in a single RUN command to avoid duplicating all of the
# files across image layers when the permissions change
RUN conda install --quiet --yes \
    'jupyter=1.0.0' && \
    conda clean --all -f -y && \
    jupyter notebook --generate-config && \
    jupyter lab clean && \
    rm -rf /home/$NB_USER/.cache/yarn && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

EXPOSE 8888

# Configure container startup
ENTRYPOINT ["tini", "-g", "--"]
CMD ["start-notebook.sh"]

# Copy local files as late as possible to avoid cache busting
COPY scripts/start.sh \
     scripts/start-notebook.sh \
     scripts/start-singleuser.sh \
     /usr/local/bin/
     
# Currently need to have both jupyter_notebook_config and jupyter_server_config to support classic and lab
COPY config/jupyter_notebook_config.py /etc/jupyter/

# Fix permissions on /etc/jupyter as root
USER root

# Prepare upgrade to JupyterLab V3.0 #1205
RUN sed -re "s/c.NotebookApp/c.ServerApp/g" \
    /etc/jupyter/jupyter_notebook_config.py > /etc/jupyter/jupyter_server_config.py

RUN fix-permissions /etc/jupyter/

# Switch back to jovyan to avoid accidental container runs as root

WORKDIR $HOME


# [3] Install Conda packages ............. #

# Create alternative for nano -> nano-tiny
RUN update-alternatives --install /usr/bin/nano nano /bin/nano-tiny 10

RUN git clone https://github.com/Syllo/nvtop.git /run/nvtop && \
    mkdir -p /run/nvtop/build && cd /run/nvtop/build && \
    (cmake .. -DNVML_RETRIEVE_HEADER_ONLINE=True 2> /dev/null || echo "cmake was not successful") && \
    (make 2> /dev/null || echo "make was not successful") && \
    (make install 2> /dev/null || echo "make install was not successful") && \
    cd /tmp && rm -rf /tmp/nvtop

USER $NB_UID

# Install Python 3 packages
RUN conda config --add channels conda-forge && \
    conda install --quiet --yes --file=${PKG_DIR}/static-conda-packages.txt && \
    conda clean --all -f -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

# Install facets which does not have a pip or conda package at the moment
WORKDIR /tmp
RUN git clone https://github.com/PAIR-code/facets.git && \
    jupyter nbextension install facets/facets-dist/ --sys-prefix && \
    rm -rf /tmp/facets && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

# Import matplotlib the first time to build the font cache.
ENV XDG_CACHE_HOME="/home/${NB_USER}/.cache/"

RUN MPLBACKEND=Agg python -c "import matplotlib.pyplot" && \
    fix-permissions "/home/${NB_USER}"


# [4] Install pip packages ............. #

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r ${PKG_DIR}/static-pip-packages.txt

WORKDIR $HOME


# [5] Install Jupyterlab Extensions ............. #

RUN jupyter nbextension enable --py --sys-prefix ipyleaflet
RUN jupyter labextension install jupyterlab-plotly \
                                 @jupyter-widgets/jupyterlab-manager plotlywidget \
                                 @ijmbarr/jupyterlab_spellchecker \
                                 jupyterlab-theme-solarized-dark \
                                 jupyterlab-tailwind-theme \
                                 jupyterlab-theme-hale \
                                 jupyterlab_onedarkpro
RUN fix-permissions $CONDA_DIR
RUN fix-permissions /home/$NB_USER

USER $NB_UID


# [6] Install Dynamic Content ............. #

# Copy jupyter_notebook_config.json
COPY config/jupyter_notebook_config.json /etc/jupyter/

# Dyamic apt packages
USER root
COPY dynamic-elements/dynamic-apt-packages.txt ${PKG_DIR}
RUN apt-get update && \
    xargs -a ${PKG_DIR}/dynamic-apt-packages.txt \
    apt-get install -yq --no-install-recommends \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
USER $NB_UID

# Dynamic pip packages
COPY dynamic-elements/dynamic-pip-packages.txt ${PKG_DIR}
RUN pip install --no-cache-dir -r ${PKG_DIR}/dynamic-pip-packages.txt

# Copy custom config settings
COPY config/lab /home/jovyan/.jupyter/lab