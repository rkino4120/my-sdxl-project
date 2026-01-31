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

# .envファイルから環境変数を読み込む
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ .envファイルから設定を読み込みました")
except ImportError:
    pass

ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")
API_KEY = os.getenv("RUNPOD_API_KEY")

if not ENDPOINT_ID or not API_KEY:
    raise ValueError("環境変数 RUNPOD_ENDPOINT_ID と RUNPOD_API_KEY を設定してください。")

def contains_japanese(text):
    """テキストに日本語が含まれているかチェック"""
    return bool(re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', text))

def translate_to_english(text):
    """日本語を英語に翻訳"""
    if not TRANSLATOR_AVAILABLE or not contains_japanese(text):
        return text
    
    try:
        print(f"📝 日本語プロンプトを検出: {text}")
        translated = GoogleTranslator(source='ja', target='en').translate(text)
        print(f"✓ 英語に翻訳: {translated}")
        return translated
    except Exception as e:
        print(f"⚠️  翻訳エラー: {e}")
        return text

def encode_image_to_base64(image_path):
    """画像をBase64エンコード"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# ==========================================
# IP-Adapter使用例：参照画像から人物の特徴を抽出
# ==========================================

url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

# 参照画像のパス（存在する場合のみ使用）
reference_image_path = "taiwanese01.png"  # ここに参照画像のパスを指定

# プロンプト
prompt = "富士山の前に立つ人物、美しい風景、8K高画質"
negative_prompt = "低品質、ぼやけた"

# 日本語を英語に翻訳
prompt_en = translate_to_english(prompt)
negative_prompt_en = translate_to_english(negative_prompt)

payload = {
    "input": {
        "prompt": prompt_en,
        "negative_prompt": negative_prompt_en,
        "steps": 30,
        "guidance_scale": 7.5,
        "width": 2048,
        "height": 2048,
        "ip_adapter_scale": 0.6  # 参照画像の影響度 (0.0-1.0)
    }
}

# 参照画像が存在する場合は追加
if os.path.exists(reference_image_path):
    print(f"📸 参照画像を読み込み: {reference_image_path}")
    payload["input"]["reference_image"] = encode_image_to_base64(reference_image_path)
    print("✓ 参照画像をエンコード完了")
else:
    print("⚠️  参照画像が見つかりません。通常のtext-to-imageで生成します。")
    print(f"   参照画像を使う場合: {reference_image_path} に画像を配置してください。")

print("\nリクエスト送信中...")
start_time = time.time()

try:
    response = requests.post(url, json=payload, headers=headers, timeout=600)
    response_data = response.json()
    
    print(f"完了！ かかった時間: {time.time() - start_time:.2f}秒")

    if 'output' in response_data:
        output = response_data['output']
        
        if 'error' in output:
            print(f"サーバーエラー: {output['error']}")
        elif 'image' in output:
            img_base64 = output['image']
            
            # 画像保存
            image = Image.open(BytesIO(base64.b64decode(img_base64)))
            output_filename = "output_ip_adapter.png" if os.path.exists(reference_image_path) else "output_runpod.png"
            image.save(output_filename)
            
            print(f"\n✓ 画像保存完了: {output_filename}")
            print(f"  プロンプト: {output.get('prompt', 'N/A')}")
            print(f"  サイズ: {output.get('width', 'N/A')}x{output.get('height', 'N/A')}")
            print(f"  ステップ数: {output.get('steps', 'N/A')}")
            
            if os.path.exists(reference_image_path):
                print(f"  参照画像使用: はい (影響度: {payload['input']['ip_adapter_scale']})")
        else:
            print("予期せぬレスポンス形式:")
            print(output)
    else:
        print("エラーまたは予期せぬレスポンス:")
        print(response_data)

except Exception as e:
    print(f"通信エラーが発生しました: {e}")
