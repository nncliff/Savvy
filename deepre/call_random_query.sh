#!/bin/bash

# Call random query endpoint
LOG_FILE="/app/logs/random_query_cron.log"
mkdir -p "$(dirname "$LOG_FILE")"
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S JST")

echo "[$TIMESTAMP] Starting random query request..." >> "$LOG_FILE"

# Iterate through each subdirectory in user_storage
for user_dir in /app/user_storage/*/; do
    if [ -d "$user_dir" ]; then
        user_folder=$(basename "$user_dir")
        echo "[$TIMESTAMP] Processing user folder: $user_folder" >> "$LOG_FILE"
        
        curl -X POST "http://localhost:8000/random-query" \
             -H "Content-Type: application/json" \
             -d "{\"method\": \"global\", \"root\": \"$user_dir\"}" \
             -w "\nHTTP Status: %{http_code}\nTotal Time: %{time_total}s\n" \
             >> "$LOG_FILE" 2>&1
        
        if [ $? -eq 0 ]; then
            echo "[$TIMESTAMP] Random query request for $user_folder completed successfully" >> "$LOG_FILE"
        else
            echo "[$TIMESTAMP] Random query request for $user_folder failed" >> "$LOG_FILE"
        fi
        echo "--- End of $user_folder ---" >> "$LOG_FILE"
    fi
done

# Check if no subdirectories were found
if [ ! -d "/app/user_storage/"* ]; then
    echo "[$TIMESTAMP] No user subdirectories found in user_storage" >> "$LOG_FILE"
fi

echo "----------------------------------------" >> "$LOG_FILE"
