FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    git \
    vim \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel

RUN mkdir -p /usr/src
WORKDIR /usr/src
RUN apt-get install wget
RUN apt-get install zip unzip
RUN wget https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get install git-lfs
RUN pip3 install tensorflow


COPY ./requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt
WORKDIR /usr/
RUN wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
RUN unzip wiki-news-300d-1M.vec.zip
WORKDIR /usr/src/