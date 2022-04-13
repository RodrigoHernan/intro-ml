FROM python:3.8-bullseye

ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install -y software-properties-common

RUN apt-get update && apt-get install -y --no-install-recommends \
    default-jdk curl -y

COPY ./requirements.txt /requirements.txt
RUN pip install -r requirements.txt

EXPOSE 8888

WORKDIR /app
