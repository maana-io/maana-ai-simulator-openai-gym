FROM tiangolo/uvicorn-gunicorn:python3.7

RUN pip install uvicorn gunicorn ariadne graphqlclient asgi-lifespan
RUN pip install gym

COPY ./app /app
COPY start.sh /start.sh
WORKDIR /