FROM python:3.9.20-slim-bookworm

RUN apt-get update &&\
    apt-get install -y --no-install-recommends graphviz &&\
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /ConstrainTree/requirements.txt
COPY execute.py /ConstrainTree/execute.py

WORKDIR /ConstrainTree

RUN python -m pip install --no-cache-dir -r requirements.txt

# keep the container running
ENTRYPOINT ["tail", "-f", "/dev/null"]
