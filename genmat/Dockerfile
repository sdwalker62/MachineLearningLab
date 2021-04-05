# set base image (host OS)
FROM python:3.9

# copy the dependencies file to the working directory
COPY requirements.txt requirements.txt

# install dependencies
RUN pip install -r requirements.txt

WORKDIR /code

RUN mkdir -p code/database
RUN mkdir -p code/results

COPY code/ ./genmat
COPY mod_drain ./genmat/mod_drain

# command to run on container start
RUN ls
CMD [ "python", "-u", "-m", "genmat" ]