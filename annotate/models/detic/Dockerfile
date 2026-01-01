FROM annotation-base:latest

# Install detectron2
RUN pip install --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git'

# Clone Detic and CenterNet2
RUN git clone --recurse-submodules https://github.com/facebookresearch/Detic.git /app/Detic \
    && cd /app/Detic \
    && pip install --no-cache-dir -r requirements.txt

# Add Detic and CenterNet2 to Python path
ENV PYTHONPATH="/app/Detic:/app/Detic/third_party/CenterNet2:${PYTHONPATH}"

COPY models/detic/requirements.txt /tmp/model_requirements.txt
RUN pip install --no-cache-dir -r /tmp/model_requirements.txt

# Symlink Detic datasets so relative paths resolve from /app
RUN ln -s /app/Detic/datasets /app/datasets

COPY models/detic/app.py /app/app.py
COPY models/detic/model.py /app/model.py

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
