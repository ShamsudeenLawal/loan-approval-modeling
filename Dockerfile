FROM python:3.11-slim

WORKDIR /code

COPY requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir -r /code/requirements.txt

COPY utils /code/utils

COPY static /code/static

COPY templates /code/templates

COPY models /code/models

COPY app.py /code/app.py

CMD [ "python", "/code/app.py" ]
