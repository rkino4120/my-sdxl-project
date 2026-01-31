# SDXL Image Generation - RunPod Serverless

RunPod Serverless GPUでStable Diffusion XLによる画像生成を行うプロジェクト

## 📁 ファイル構成

```
my-sdxl-project/
├── handler.py              # RunPod Serverless ハンドラー（サーバー側）
├── client.py               # APIクライアント（ローカル実行用）
├── Dockerfile              # RunPodデプロイ用Dockerイメージ
├── requirements.txt        # サーバー側の依存関係
├── requirements-dev.txt    # 開発環境の依存関係
├── .env.example           # 環境変数テンプレート
└── .gitignore             # Git除外設定
```

## 🚀 セットアップ

### ローカル開発環境

```powershell
# 仮想環境の作成と有効化
python -m venv .venv
.\.venv\Scripts\activate

# 開発用パッケージのインストール
pip install -r requirements-dev.txt

# 環境変数の設定
Copy-Item .env.example .env
notepad .env  # 実際のIDとAPIキーを記入
```

### RunPodデプロイ

```powershell
# Dockerイメージのビルド
docker build -t my-sdxl-serverless .

# Docker Hubにプッシュ
docker tag my-sdxl-serverless <your-dockerhub-username>/my-sdxl-serverless:latest
docker push <your-dockerhub-username>/my-sdxl-serverless:latest
```

## 🎨 使い方

### クライアントから画像生成

```powershell
# .envファイルに認証情報を設定後
python client.py
```

### API入力形式

```json
{
  "input": {
    "prompt": "a beautiful sunset over mountains",
    "negative_prompt": "low quality, blurry",
    "steps": 30,
    "guidance_scale": 7.5,
    "width": 1024,
    "height": 1024,
    "seed": 42
  }
}
```

### API出力形式

```json
{
  "image": "<base64-encoded-png>",
  "prompt": "...",
  "seed": 42,
  "steps": 30,
  "width": 1024,
  "height": 1024
}
```

## ⚙️ パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| prompt | - | 生成したい画像の説明（必須） |
| negative_prompt | "low quality, blurry" | 除外したい要素 |
| steps | 30 | 推論ステップ数（多いほど高品質だが時間がかかる） |
| guidance_scale | 7.5 | プロンプトへの忠実度 |
| width | 1024 | 画像の幅 |
| height | 1024 | 画像の高さ |
| seed | None | 再現性のためのシード値 |

## 🔧 トラブルシューティング

### 環境変数エラー

```
ValueError: 環境変数 RUNPOD_ENDPOINT_ID と RUNPOD_API_KEY を設定してください。
```

→ `.env`ファイルを作成するか、環境変数を設定してください

### Docker ビルドエラー

- CUDA対応GPUが必要です
- ベースイメージのダウンロードに時間がかかります
- モデルの事前ダウンロードでイメージサイズが約13GBになります

## 📝 注意事項

- RunPod Serverlessのペイロード制限: `/runsync` は20MB
- 大きな画像（例: 2048x2048）はBase64エンコード後に制限を超える可能性があります
- 商用利用時は透かしを無効化しています（handler.py参照）

## 🔐 セキュリティ

- `.env`ファイルは`.gitignore`で除外されています
- APIキーは絶対にGitにコミットしないでください
- 環境変数または`.env`ファイルを使用してください

## 📚 参考リンク

- [RunPod Documentation](https://docs.runpod.io/)
- [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- [Diffusers Library](https://huggingface.co/docs/diffusers/)
