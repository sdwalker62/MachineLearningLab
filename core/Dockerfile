# set base image (host OS)
FROM python:3.8.6

# copy the dependencies file to the working directory
COPY requirements.txt requirements.txt

# install dependencies
RUN pip install -r requirements.txt

WORKDIR /LAT

RUN mkdir -p LAT/database
RUN mkdir -p LAT/results
RUN mkdir -p LAT/ancillary

COPY /ancillary ./ancillary
COPY /training ./training
COPY drain3.ini /LAT/

# command to run on container start
CMD [ "python", "-u", "-m", "training" ]