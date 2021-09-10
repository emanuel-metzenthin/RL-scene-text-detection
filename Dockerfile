FROM rayproject/ray:1.3.0-py38-gpu

USER root

RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6 gcc -y

RUN mkdir /app
COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt
RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
