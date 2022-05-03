## Dockerによる環境構築

Dockerおよび[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)をインストールしてあるUbuntuを前提とします。

1. Dockerfileをダウンロードする
```shell
wget https://raw.githubusercontent.com/SakodaShintaro/nlp_practice/master/docker/Dockerfile
```

2. nlp_practiceというイメージを作成する
```shell
docker build -t nlp_practice:latest .
```

3. nlp_practiceをもとにnlp_practice_containerというコンテナを作成する
```shell
docker run --gpus all -it --name nlp_practice_container nlp_practice:latest bash
```
