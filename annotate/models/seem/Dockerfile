FROM annotation-base:latest

# Install system dependencies for mpi4py
RUN apt-get update && apt-get install -y \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install detectron2 (required by SEEM's xdecoder)
RUN pip install --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git'

# Clone SEEM repo for model code
RUN git clone https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once.git /app/SEEM

# GPU available - no need to patch CUDA references

# Install git-based dependencies that SEEM needs
RUN pip install --no-cache-dir \
    'git+https://github.com/cocodataset/panopticapi.git' \
    'git+https://github.com/openai/CLIP.git'

ENV PYTHONPATH="/app/SEEM:${PYTHONPATH}"

COPY models/seem/requirements.txt /tmp/model_requirements.txt
RUN pip install --no-cache-dir -r /tmp/model_requirements.txt

COPY models/seem/app.py /app/app.py
COPY models/seem/model.py /app/model.py

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
