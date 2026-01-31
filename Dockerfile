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
# 初回は無効化して問題を特定しやすくする
# 注意: モデルは初回起動時にダウンロードされます（起動に数分かかります）
# RunPod Network Volumeを使用することで2回目以降は高速化されます

# RunPod Serverless起動
CMD ["python3", "-u", "handler.py"]