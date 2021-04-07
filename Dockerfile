FROM ubuntu:18.04
LABEL maintainer="Guilherme Bauer Negrini <negrini.guilherme@gmail.com>"
LABEL description="Tolkien character or prescription drug"

RUN apt update -y && \
    apt-get -y install software-properties-common && \
    apt-get -y install wget && \
    apt-get install -y python3 python3-pip && \
    apt clean

RUN ["mkdir", "tolkien-char-prescription-drug"]

COPY . /tolkien-char-prescription-drug
COPY configs/.jupyter /root/.jupyter

WORKDIR /tolkien-char-prescription-drug

RUN python3 -m pip install --upgrade pip
RUN pip3 install -r requirements.txt

EXPOSE 8888

