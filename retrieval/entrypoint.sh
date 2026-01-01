#!/bin/bash
set -e
python etl.py
exec uvicorn api_server:app --host 0.0.0.0 --port 8000
