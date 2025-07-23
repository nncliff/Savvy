#!/bin/bash

# Call random query endpoint
LOG_FILE="/app/logs/random_query_cron.log"
mkdir -p "$(dirname "$LOG_FILE")"
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S JST")

echo "[$TIMESTAMP] Starting random query request..." >> "$LOG_FILE"

curl -X POST "http://localhost:8000/random-query" \
     -H "Content-Type: application/json" \
     -d '{"method": "global", "root": "./"}' \
     -w "\nHTTP Status: %{http_code}\nTotal Time: %{time_total}s\n" \
     >> "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
    echo "[$TIMESTAMP] Random query request completed successfully" >> "$LOG_FILE"
else
    echo "[$TIMESTAMP] Random query request failed" >> "$LOG_FILE"
fi

echo "----------------------------------------" >> "$LOG_FILE"