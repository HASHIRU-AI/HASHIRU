# HASHIRU v2 in one container: the chat page, OpenAI-compatible API, A2A agent and MCP server on one port.
# Hugging Face Docker Space: add Space secrets HASHIRU_API_KEY (required: the hub refuses to listen publicly
# without it), CEO_MODEL, MODELS and the provider key those models need (e.g. GEMINI_API_KEY). A Space has no
# Ollama, so use cloud models; elsewhere, OLLAMA_URL can point at one.
FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.6 /uv /uvx /bin/
RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
    HASHIRU_FRAME_ANCESTORS="https://huggingface.co"
WORKDIR /home/user/app

COPY --chown=user pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project
COPY --chown=user src ./src
RUN uv sync --locked --no-dev

EXPOSE 7860
# Hugging Face sets SPACE_HOST; elsewhere set HASHIRU_PUBLIC_URL to the address clients use.
CMD ["sh", "-c", "exec .venv/bin/hashiru serve host=0.0.0.0 port=7860 ${SPACE_HOST:+public_url=https://$SPACE_HOST}"]
