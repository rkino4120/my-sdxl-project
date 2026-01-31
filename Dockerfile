# RunPod推奨のベースイメージ（PyTorch + CUDA）
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# 環境変数の設定
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/huggingface-cache

WORKDIR /app

# システムパッケージの更新と必要なツールのインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 必要なPythonライブラリをインストール
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ハンドラーコードをコピー
COPY handler.py .

# （オプション）モデルを事前ダウンロードして起動を高速化
# 注意: これによりイメージサイズが大きくなります（約13GB）
# 本番環境では RunPod Network Volume にキャッシュすることを推奨
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download('stabilityai/stable-diffusion-xl-base-1.0', \
    ignore_patterns=['*.ckpt', '*.safetensors'], allow_patterns=['*.json', '*.txt']); \
    snapshot_download('madebyollin/sdxl-vae-fp16-fix')"

# RunPod Serverless起動
CMD ["python3", "-u", "handler.py"]