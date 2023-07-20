# This is the DockerFile used for YOLO NAS System #
# This environment make use of Ubuntu20.04 with Cuda 11.1  #

FROM nvidia/cuda:11.1.1-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install python version of 3.8
RUN apt-get update
RUN apt update -y && apt upgrade -y && \
    apt-get install -y wget build-essential checkinstall  libreadline-gplv2-dev  libncursesw5-dev  libssl-dev  libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev && \
    cd /usr/src && \
    wget https://www.python.org/ftp/python/3.8.10/Python-3.8.10.tgz && \
    tar xzf Python-3.8.10.tgz && \
    cd Python-3.8.10 && \
    apt-get install liblzma-dev && \
    apt-get install lzma && \
    ./configure --enable-optimizations && \
    make altinstall
# link python to python 3.8 version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.8 1

# install libraries for opencv
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
# install libraries for git and curl
RUN apt-get install git curl -y

# install system libraries from requirements.txt
COPY requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache \ 
    python3 -m pip install -r requirements.txt