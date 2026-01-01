FROM annotation-base:latest

COPY models/grounding_dino/requirements.txt /tmp/model_requirements.txt
RUN pip install --no-cache-dir -r /tmp/model_requirements.txt

COPY models/grounding_dino/app.py /app/app.py
COPY models/grounding_dino/model.py /app/model.py

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
