#!/bin/bash

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Check if Docker Compose is installed
if ! command_exists docker-compose; then
  echo "Docker Compose is not installed. Installing Docker Compose..."
  
  # Download and install Docker Compose
  sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
  
  # Make Docker Compose executable
  sudo chmod +x /usr/local/bin/docker-compose
  
  # Verify installation
  if command_exists docker-compose; then
    echo "Docker Compose installed successfully."
  else
    echo "Error: Failed to install Docker Compose."
    exit 1
  fi
else
  echo "Docker Compose is already installed."
fi

# Check if docker-compose.yml exists
if [ ! -f docker-compose.yml ]; then
  echo "Error: docker-compose.yml not found!"
  exit 1
fi

# Build Docker images
echo "Building Docker images..."
docker compose up --build -d

if [ $? -eq 0 ]; then
  echo "Docker images built successfully."
  docker ps
else
  echo "Error: Failed to build Docker images."
  exit 1
fi