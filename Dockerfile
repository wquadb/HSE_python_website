FROM python:3.10-slim

WORKDIR /app

RUN pip3 install gunicorn

COPY requirments.txt .
RUN pip3 install --default-timeout=100 -r requirements.txt

COPY . .