#!/bin/bash

# Check if docker-compose.yml exists
if [ ! -f docker-compose.yml ]; then
    echo "Error: docker-compose.yml not found! - Intalling..."
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# Build Docker images
echo "Building Docker images..."
docker-compose build -d

if [ $? -eq 0 ]; then
    echo "Docker images built successfully."
    docker ps
else
    echo "Error: Failed to build Docker images."
    exit 1
fi