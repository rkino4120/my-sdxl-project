import requests
import base64
from PIL import Image
from io import BytesIO
import time
import os
from pathlib import Path
import re

# 日本語→英語翻訳（オプション）
try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False
    print("⚠️  deep-translatorがインストールされていません。日本語プロンプトは英語に翻訳されません。")
    print("   インストール: pip install deep-translator")

# .envファイルから環境変数を読み込む（python-dotenvがある場合）
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ .envファイルから設定を読み込みました")
except ImportError:
    pass  # python-dotenvがない場合はスキップ

# ==========================================
# 環境変数から認証情報を取得（必須）
# 使い方:
#   方法1: .envファイルを使う（推奨）
#     1. .env.example を .env にコピー
#     2. .env に実際の値を記入
#     3. pip install python-dotenv
#
#   方法2: 環境変数を直接設定
#     PowerShell: $env:RUNPOD_ENDPOINT_ID = "your_id"
#                 $env:RUNPOD_API_KEY = "your_key"
#     Linux/Mac:  export RUNPOD_ENDPOINT_ID="your_id"
#                 export RUNPOD_API_KEY="your_key"
# ==========================================
ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")
API_KEY = os.getenv("RUNPOD_API_KEY")

if not ENDPOINT_ID or not API_KEY:
    raise ValueError(
        "❌ 環境変数 RUNPOD_ENDPOINT_ID と RUNPOD_API_KEY を設定してください。\n\n"
        "【推奨】.envファイルを使う:\n"
        "  1. .env.example を .env にコピー\n"
        "  2. .env に実際のIDとAPIキーを記入\n"
        "  3. pip install python-dotenv を実行\n\n"
        "【代替】環境変数を直接設定:\n"
        "  PowerShell: $env:RUNPOD_ENDPOINT_ID = 'your_id'; $env:RUNPOD_API_KEY = 'your_key'\n"
    )
# ==========================================

def contains_japanese(text):
    """テキストに日本語が含まれているかチェック"""
    return bool(re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', text))

def translate_to_english(text):
    """日本語を英語に翻訳（日本語が含まれている場合のみ）"""
    if not TRANSLATOR_AVAILABLE:
        return text
    
    if contains_japanese(text):
        try:
            print(f"📝 日本語プロンプトを検出: {text}")
            translated = GoogleTranslator(source='ja', target='en').translate(text)
            print(f"✓ 英語に翻訳: {translated}")
            return translated
        except Exception as e:
            print(f"⚠️  翻訳エラー（元のテキストを使用）: {e}")
            return text
    return text

# RunPodのエンドポイント
# runsyncは便利ですが、長引くとタイムアウトするので、その対策を入れます
url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync"
status_url_template = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/status/"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

# プロンプト（日本語OK - 自動的に英語に翻訳されます）
prompt = "富士山の夕焼け、美しい風景、8K高画質、masterpiece"
negative_prompt = "低品質、ぼやけた、テキスト、透かし"

# 日本語が含まれていれば自動翻訳
prompt_en = translate_to_english(prompt)
negative_prompt_en = translate_to_english(negative_prompt)

payload = {
    "input": {
        "prompt": prompt_en,
        "negative_prompt": "low quality, worst quality, blurry, text, watermark",
        "steps": 30,
        "guidance_scale": 7.5,
        "seed": 42,
        "width": 1024,
        "height": 1024
    }
}

def decode_and_save_image(img_data):
    """画像データをデコードして保存する関数"""
    try:
        image = Image.open(BytesIO(base64.b64decode(img_data)))
        image.save("output_runpod.png")
        print("\n✅ 画像保存完了: output_runpod.png を確認してください！")
    except Exception as e:
        print(f"画像の保存に失敗しました: {e}")

print("🚀 リクエスト送信中...（サーバー起動待ちの場合、数分かかります）")
start_time = time.time()

try:
    # 1. まず生成リクエストを送る
    response = requests.post(url, json=payload, headers=headers, timeout=600)
    response_data = response.json()
    
    status = response_data.get('status')
    job_id = response_data.get('id')

    print(f"初期ステータス: {status} (ID: {job_id})")

    # 2. まだ終わっていない場合(IN_QUEUE または IN_PROGRESS)は、終わるまで監視する
    if status in ['IN_QUEUE', 'IN_PROGRESS'] and job_id:
        print("⏳ 処理待ちまたは実行中... 完了まで定期的に確認します。")
        
        while True:
            time.sleep(5) # 5秒待機
            
            # ステータス確認APIを叩く
            check_url = status_url_template + job_id
            check_res = requests.get(check_url, headers=headers)
            check_data = check_res.json()
            
            current_status = check_data.get('status')
            print(f"\r経過時間: {time.time() - start_time:.1f}秒 - 現在の状況: {current_status}", end="")
            
            if current_status == 'COMPLETED':
                response_data = check_data # データを上書き
                print("\n✨ 生成完了！")
                break
            elif current_status == 'FAILED':
                print("\n❌ 生成失敗。")
                print("エラー詳細:", check_data)
                exit()
            
            # まだならループ継続

    # 3. 結果の取り出し（完了時）
    if 'output' in response_data:
        output = response_data['output']
        
        # app.pyの返し方によって構造が違う場合への対応
        img_base64 = None
        
        # パターンA: { "output": "base64..." }
        if isinstance(output, dict) and 'output' in output: 
             img_base64 = output['output']
        # パターンB: { "image": "base64..." }
        elif isinstance(output, dict) and 'image' in output:
             img_base64 = output['image']
        # パターンC: 単なる文字列として返ってくる場合
        elif isinstance(output, str):
             img_base64 = output
             
        if img_base64:
            decode_and_save_image(img_base64)
        else:
            print("\n⚠️ 画像データが見つかりませんでした。レスポンスの中身:")
            print(response_data)
            
    else:
        print("\nエラーまたは予期せぬレスポンス:")
        print(response_data)

except Exception as e:
    print(f"\n通信エラーが発生しました: {e}")