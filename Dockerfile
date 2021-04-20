FROM pytorch/pytorch

#RUN mkdir /app && chown -R $UID:$GID /app && chmod -R 700 /app
COPY requirements.txt /app/requirements.txt
COPY . /app/

WORKDIR /app
RUN pip install -r requirements.txt