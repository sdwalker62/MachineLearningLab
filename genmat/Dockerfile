# set base image (host OS)
FROM python:3.9

MAINTAINER Dalton Walker "s.d.walker6249@gmail.com"

# copy the dependencies file to the working directory
COPY requirements.txt requirements.txt

# install dependencies
RUN pip install -r requirements.txt

WORKDIR /code

RUN mkdir -p code/database
RUN mkdir -p code/results

COPY code/ ./genmat

# command to run on container start
CMD [ "python", "-u", "-m", "genmat" ]