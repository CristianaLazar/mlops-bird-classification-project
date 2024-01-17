FROM python:3.9

WORKDIR /code
COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY ./app /code/app
COPY ./models /code/models

CMD exec uvicorn app.application:app --port $PORT --host 0.0.0.0 --workers 1