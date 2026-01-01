ARG CUDA_BASE_IMAGE=nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
FROM ${CUDA_BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ARG TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ARG PYTORCH_VERSION=2.5.1
ARG TORCHVISION_VERSION=0.20.1
ARG PYTORCH_PACKAGES="torch==2.5.1 torchvision==0.20.1"
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
ARG INSTALL_TORCH=1

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    git wget libgl1 libglib2.0-0 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CUDA-enabled PyTorch unless the base image already provides it.
RUN if [ "${INSTALL_TORCH}" = "1" ]; then \
        if [ -n "${PYTORCH_INDEX_URL}" ]; then \
            pip install --no-cache-dir ${PYTORCH_PACKAGES} --index-url ${PYTORCH_INDEX_URL}; \
        else \
            pip install --no-cache-dir ${PYTORCH_PACKAGES}; \
        fi; \
    else \
        python -c "import importlib.util; missing=[name for name in ('torch','torchvision') if importlib.util.find_spec(name) is None]; assert not missing, 'Missing required preinstalled packages in base image: ' + ', '.join(missing); print('Using preinstalled torch/torchvision from base image')"; \
    fi

# Install common dependencies
COPY models/base/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy shared package
COPY shared/ /app/shared/
