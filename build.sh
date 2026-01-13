#!/bin/bash
set -e

IMAGE="registry.local.jbmurphy.com/mcp-soundcloud:latest"

echo "Building mcp-soundcloud for linux/amd64..."
docker buildx build --platform linux/amd64 -t "$IMAGE" --push .

echo "Done. Image pushed to $IMAGE"
