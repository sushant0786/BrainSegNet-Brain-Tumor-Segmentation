FROM python:3.11-slim
WORKDIR /app

# ── install deps first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── fetch the checkpoint exactly once, while building
RUN pip install --no-cache-dir huggingface_hub && \
    python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="TejasB5/unet_brain_tumor",
    filename="unet_best.pth",
    cache_dir="/app/weights",            
    force_download=True
)
PY


EXPOSE 8000
CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
