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

# Copy package files and install as node user
COPY --chown=node:node graphrag-visualizer/package*.json ./graphrag-visualizer/
WORKDIR /app/graphrag-visualizer
USER node
RUN npm install

# Switch back to root for copying files and creating symlink
USER root
WORKDIR /app
COPY . .

# Create directory structure and symbolic link
RUN ln -sf /app/docker_volume/zhan.chen_gmail.com/output /app/graphrag-visualizer/public/artifacts

# Fix ownership of all files
RUN chown -R node:node /app

# Switch to node user
USER node
WORKDIR /app/graphrag-visualizer

# Set environment variable for port
ENV PORT=3030

# Expose port
EXPOSE 3030

# Set default command
CMD ["npm", "start"]
#CMD ["/bin/bash"]
