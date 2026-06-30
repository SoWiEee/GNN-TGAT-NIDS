FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock* LICENSE README.md ./
RUN uv python install 3.12
RUN uv sync --no-dev --no-install-project --python 3.12

RUN uv pip install --no-cache-dir \
    pyg_lib torch_scatter torch_sparse torch_cluster \
    -f https://data.pyg.org/whl/torch-2.6.0+cu124.html

COPY src/ src/
COPY app/ app/
COPY configs/ configs/
COPY data/demo/ data/demo/
COPY data/metrics/ data/metrics/
COPY train.py attack.py ./

RUN mkdir -p data/raw data/processed data/sessions checkpoints

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
