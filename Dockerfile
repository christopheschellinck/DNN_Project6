FROM ubuntu:18.04
FROM python:3.8

RUN apt-get update -y 
RUN pip3 install -U scikit-learn

RUN mkdir /app 
COPY requirements.txt .
COPY app_CNN.py /app/app_CNN.py
COPY asset /app/asset
COPY mobile_net_test.h5 /app/mobile_net_test.h5

RUN pip install -r requirements.txt

WORKDIR /app

CMD ["python", "app_CNN.py"]


