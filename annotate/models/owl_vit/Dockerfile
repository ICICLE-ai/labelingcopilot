FROM annotation-base:latest

# OWL-ViT only needs transformers (already pure HuggingFace)
COPY models/owl_vit/requirements.txt /tmp/model_requirements.txt
RUN pip install --no-cache-dir -r /tmp/model_requirements.txt

COPY models/owl_vit/app.py /app/app.py
COPY models/owl_vit/model.py /app/model.py

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
