# set base image (host OS)
FROM python:3.8.6

# copy the dependencies file to the working directory
COPY requirements.txt requirements.txt

# install dependencies
RUN /usr/local/bin/python -m pip install --upgrade pip
RUN pip install -r requirements.txt

WORKDIR /training

RUN mkdir -p training/results

COPY __main__.py .
COPY LogSequenceEncoder.py .
COPY Metrics.py .
COPY MultiHeadAttention.py .
COPY PositionalEncoder.py .
COPY Transformer.py .
COPY TransformerBlock.py .

# command to run on container start
CMD [ "python", "-u", "__main__.py" ]