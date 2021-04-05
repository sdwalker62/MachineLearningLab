# set base image (host OS)
FROM python:3.8.6

# copy the dependencies file to the working directory
COPY requirements.txt requirements.txt

# install dependencies
RUN pip install -r requirements.txt

WORKDIR /training

RUN mkdir -p training/database
RUN mkdir -p training/results

COPY . .

# command to run on container start
CMD [ "python", "-u", "__main__.py" ]