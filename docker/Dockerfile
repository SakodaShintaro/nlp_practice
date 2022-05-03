FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu20.04

# GPG errorが出る問題への対処
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub

# apt更新
RUN apt-get update

# 言語の設定
RUN apt-get install -y language-pack-ja-base language-pack-ja
ENV LANG='ja_JP.UTF-8'

# 必要なもののインストール
RUN apt-get install sudo -y \
 && apt-get install git -y \
 && apt-get install vim -y \
 && apt-get install python3 -y \
 && apt-get install python3-pip -y

RUN pip3 install torch
RUN pip3 install transformers
RUN pip3 install fugashi
RUN pip3 install ipadic
RUN pip3 install autopep8

# WORKDIRを設定
WORKDIR /root

# dotfileの導入
RUN git clone https://github.com/SakodaShintaro/dotfiles && ./dotfiles/setup.sh

# このリポジトリのclone
RUN git clone https://github.com/SakodaShintaro/nlp_practice
