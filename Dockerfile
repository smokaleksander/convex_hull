FROM python:3.9-slim

WORKDIR /src

COPY . .

RUN pip install pipenv
RUN pipenv sync

EXPOSE 8003

CMD pipenv run uvicorn main:app --host 0.0.0.0 --port 8003
