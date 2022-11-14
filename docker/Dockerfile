#FROM ubuntu:latest
FROM python:3.8.1
COPY ./*.py /exp/
COPY ./requirements.txt /exp/requirements.txt
COPY ./api/app.py /exp/app.py
COPY ./svm_gamma=0.001_C=0.5.joblib /exp/./svm_gamma=0.001_C=0.5.joblib
RUN pip3 install --no-cache-dir -r /exp/requirements.txt
WORKDIR /exp
EXPOSE 5000
CMD ["python", "./app.py"]