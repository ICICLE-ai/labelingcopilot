FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY orchestrator/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy shared package
COPY shared/ /app/shared/

COPY orchestrator/app.py /app/app.py

EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
