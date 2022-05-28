FROM python:3.9-slim

WORKDIR /src

COPY . .

RUN pip install pipenv
RUN pipenv sync

EXPOSE 8003

CMD pipenv run python main.py
