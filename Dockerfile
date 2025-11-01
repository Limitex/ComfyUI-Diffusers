FROM alpine/git:latest AS builder
WORKDIR /app
RUN git clone --depth 1 --branch v0.3.67 https://github.com/comfyanonymous/ComfyUI.git ComfyUI && \
    rm -rf ComfyUI/.git

FROM pytorch/pytorch:2.9.0-cuda13.0-cudnn9-runtime
WORKDIR /app
COPY --from=builder /app/ComfyUI /app/ComfyUI
WORKDIR /app/ComfyUI
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        curl \
        build-essential \
    && curl -1sLf 'https://dl.cloudsmith.io/public/task/task/setup.deb.sh' | bash \
    && curl -sSL 'https://install.python-poetry.org' | python - \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get remove -y build-essential curl \
    && apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*
COPY ./requirements.txt ./tmp.txt
RUN pip install --no-cache-dir -r tmp.txt && rm tmp.txt
EXPOSE 8188
CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188"]
