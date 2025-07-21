FROM mcr.microsoft.com/playwright:v1.54.0-noble

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies in virtual environment
RUN /opt/venv/bin/pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    pydantic==2.5.0 \
    python-multipart==0.0.6 \
    playwright==1.53.0

# Install Playwright browsers
RUN /opt/venv/bin/playwright install chromium

# Copy application code
COPY main.py .

# Use the existing playwright user (pwuser) or skip custom user creation
# The playwright image already has a non-root user
USER pwuser

# Expose port
EXPOSE 8000

# Start the application with virtual environment
CMD ["/opt/venv/bin/python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]