#!/bin/bash

# Start cron service
service cron start

# Start FastAPI application
uvicorn main:app --host 0.0.0.0 --port 8000