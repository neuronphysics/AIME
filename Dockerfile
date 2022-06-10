FROM pytorch/pytorch:latest


CMD ["bash"]

WORKDIR /src
COPY ./requirements.txt .



RUN apt update
RUN apt-get update
RUN apt-get install -y git
RUN pip3 install -r requirements.txt
RUN apt install -y xvfb

