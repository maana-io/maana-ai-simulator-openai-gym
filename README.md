# OpenAI Gym Learning Environment for Maana Q

- Uses the [Python (Ariadne) Maana Q Knowledge Service](https://github.com/maana-io/q-template-service-python-ariadne) template
- Uses the [Python-SC2](https://github.com/Dentosal/python-sc2) library for communicating with the game engine, as it provides a higher-level interface that doesn't require visual interpretation of the game state
- Containerization is done using the [Uvicorn+Gunicorn Docker](https://github.com/tiangolo/uvicorn-gunicorn-docker) base image

## Build

```
pip install uvicorn gunicorn ariadne graphqlclient asgi-lifespan
pip install gym gym-retro
```

## Containerize

Then you can build your image from the directory that has your Dockerfile, e.g:

```
docker build -t my-service ./
```

## Run Debug Locally

To run the GraphQL service locally with hot reload:

```
./start-reload.sh
```

For details, please refer to the [official documentation](https://github.com/tiangolo/uvicorn-gunicorn-fastapi-docker#development-live-reload).

## Run Locally (via Docker)

To run the GraphQL service locally (Via Docker):

```
docker run -it -p 4000:80 -t my-service
```

## Run Debug Locally (via Docker)

To run the GraphQL service via Docker with hot reload:

```
docker run -it -p 4000:80 -v $(pwd):/app my-service /start-reload-docker.sh
```

For details, please refer to the [official documentation](https://github.com/tiangolo/uvicorn-gunicorn-fastapi-docker#development-live-reload).

## Deploy

```
gql mdeploy
```
