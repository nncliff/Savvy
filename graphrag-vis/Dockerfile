FROM node:20-bookworm

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    curl \
    vim \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Claude Code CLI globally
RUN npm install -g @anthropic-ai/claude-code

# Verify installations
RUN python3 --version && pip3 --version
RUN node --version && npm --version

# Set working directory
WORKDIR /app

# Copy package files first
COPY package*.json ./

# Install dependencies as root, then change ownership
RUN npm install

# Copy the rest of the application
COPY . .

# Create directory structure and symbolic link
RUN ln -sf /app/docker_volume/zhan.chen_gmail.com/output /app/public/artifacts

# Fix ownership of all files to node user
RUN chown -R node:node /app

# Switch to node user for runtime
USER node

# Set environment variable for port
ENV PORT=3030

# Expose port
EXPOSE 3030

# Set default command
CMD ["npm", "start"]
#CMD ["/bin/bash"]
