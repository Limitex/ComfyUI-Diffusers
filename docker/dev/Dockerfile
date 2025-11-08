FROM python:3.11-slim-bookworm
ENV POETRY_HOME="/opt/poetry"
ENV POETRY_VERSION="2.2.1"
ENV POETRY_VIRTUALENVS_PATH="/cache/.venv"
ENV POETRY_VIRTUALENVS_IN_PROJECT="false"
ENV PATH="$POETRY_HOME/bin:$PATH"
WORKDIR /app
RUN mkdir -p /cache/.venv && chmod 777 /cache/.venv
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        curl \
        build-essential \
        ca-certificates \
    && curl -sSL https://install.python-poetry.org | python - \
    && curl -1sLf 'https://dl.cloudsmith.io/public/task/task/setup.deb.sh' | bash \
    && apt-get update \
    && apt-get install -y task \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN poetry self add \
    poetry-plugin-export \
    poetry-plugin-shell 
RUN task --version && poetry --version
