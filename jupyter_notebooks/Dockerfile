# set base image (host OS)
FROM jupyter/tensorflow-notebook

# copy the dependencies file to the working directory
COPY requirements.txt requirements.txt

# install dependencies
RUN pip install -r requirements.txt

RUN jupyter labextension install \
    @jupyter-widgets/jupyterlab-manager 

COPY config/ /home/jovyan/.jupyter/

# install FiraCode font
COPY FiraCode-Retina.ttf /home/jovyan/.fonts
RUN fc-cache -f -v


# command to run on container start
CMD ["jupyter", "lab", "--allow-root"]
