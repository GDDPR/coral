#!/usr/bin/env bash
set -euo pipefail

NAME="coral"
IMAGE="neo4j:latest"

# Create if missing, otherwise start if stopped
if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "Container '$NAME' is already running."
  else
    echo "Starting existing container '$NAME'..."
    docker start "$NAME" >/dev/null
  fi
else
  echo "Creating and starting container '$NAME'..."
  docker run -d \
    --name "$NAME" \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/neo4jneo4j \
    -v coral_data:/data \
    -v coral_logs:/logs \
    "$IMAGE" >/dev/null
fi

echo "Done."
