#FROM ubuntu:latest
FROM python:3.8.1
COPY ./*.py /exp/
COPY ./requirements.txt /exp/requirements.txt
COPY ./main_d.py /main_d.py
COPY ./results ./results
COPY ./models ./models
RUN pip3 install -r /exp/requirements.txt
WORKDIR /exp
#CMD ["python", "main_d.py"]