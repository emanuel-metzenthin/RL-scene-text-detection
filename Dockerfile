FROM rayproject/ray-ml:1.3.0-py38-gpu

USER root

RUN mkdir /app
COPY . /app
WORKDIR /app
RUN pip install -e text-localization-environment/
RUN pip install -r requirements.txt
