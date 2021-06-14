FROM rayproject/ray:nightly-py38-gpu

USER root

RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6 gcc -y

RUN mkdir /app
COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt
