FROM ubuntu:18.04
RUN apt-get update && apt-get install -y python3 python3-pip sudo
RUN mkdir /app
COPY requirements.txt .
COPY app.py /app/app.py
COPY predictions.py /app/predictions.py
COPY mobile_net_test.h5 /app/mobile_net_test.h5
COPY /static /app/static
COPY /templates /app/templates
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt
RUN pip3 install pillow
WORKDIR /app
CMD ["python3", "app.py"]
