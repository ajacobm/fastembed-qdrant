#!/bin/bash
# Start both gRPC and HTTP servers

echo "Starting Enhanced FastEmbed gRPC server..."
uv run python src/enhanced_server.py &
GRPC_PID=$!

# Wait a bit for gRPC server to start
sleep 10

echo "Starting FastEmbed HTTP API server..."
uv run python src/http_server.py --host 0.0.0.0 --port 50052 --grpc-host localhost --grpc-port 50051 &
HTTP_PID=$!

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $?