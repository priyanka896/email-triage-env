FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx

COPY email_triage_env/pyproject.toml ./
COPY email_triage_env/ ./email_triage_env/
COPY email_triage_env/openenv.yaml ./
COPY inference.py ./
COPY requirements.txt ./
COPY start.sh ./

RUN chmod +x /app/start.sh

RUN uv venv /app/.venv && \
    . /app/.venv/bin/activate && \
    uv pip install -e "." && \
    uv pip install openai websockets

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

CMD ["/app/start.sh"]
