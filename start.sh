#!/bin/bash
# Start the environment server in the background
uvicorn email_triage_env.server.app:app --host 0.0.0.0 --port 7860 &
SERVER_PID=$!

# Wait for server to be ready
echo "Waiting for server to start..."
for i in $(seq 1 30); do
    if python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" 2>/dev/null; then
        echo "Server is ready!"
        break
    fi
    sleep 1
done

# Keep the server running forever
wait $SERVER_PID
