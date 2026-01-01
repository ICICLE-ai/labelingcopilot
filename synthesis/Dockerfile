ARG CUDA_BASE_IMAGE=nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
FROM ${CUDA_BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Install Python from the base distro to keep the image portable across
# both the CUDA 12.x Ubuntu 22.04 and CUDA 13.x Ubuntu 24.04 bases.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev python3-venv \
    git curl && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8090

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8090"]
