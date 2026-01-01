FROM annotation-base:latest

COPY models/sam/requirements.txt /tmp/model_requirements.txt
RUN pip install --no-cache-dir -r /tmp/model_requirements.txt

COPY models/sam/app.py /app/app.py
COPY models/sam/model.py /app/model.py

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
