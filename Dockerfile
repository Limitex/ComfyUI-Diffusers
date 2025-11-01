FROM pytorch/pytorch:2.9.0-cuda13.0-cudnn9-runtime AS builder
WORKDIR /app
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*
RUN git clone --branch v0.3.67 https://github.com/comfyanonymous/ComfyUI.git ComfyUI
WORKDIR /app/ComfyUI

FROM pytorch/pytorch:2.9.0-cuda13.0-cudnn9-runtime
WORKDIR /app
COPY --from=builder /app/ComfyUI /app/ComfyUI
WORKDIR /app/ComfyUI
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8188
CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188"]
