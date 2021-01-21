FROM ubuntu:18.04
RUN mkdir /app
COPY requirements.txt .
COPY app.py /app/app.py
COPY predictions.py /app/predictions.py
COPY mobile_net_test.h5 /app/mobile_net_test.h5
COPY /templates /app/templates
#RUN python3 -m pip install --upgrade pip
RUN #!/usr/bin/python3 -m pip install -r requirements.txt
RUN #!/usr/bin/pip3 install pillow
WORKDIR /app
CMD ["#!/usr/bin/python3", "app.py"]


