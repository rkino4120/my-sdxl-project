# PyTorchが含まれたベースイメージを使用
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# 必要なライブラリをインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# コードをコピー
COPY . .

# アプリケーション起動
CMD ["python3", "app.py"]