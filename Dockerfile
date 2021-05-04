FROM rayproject/ray-ml:1.3.0-py38-gpu

COPY requirements.txt /app/requirements.txt
COPY ./text-localization-environment/ /env

RUN pip install -e /env
RUN pip install -r /app/requirements.txt