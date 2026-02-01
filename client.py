import requests
import base64
from PIL import Image
from io import BytesIO
import time
import os
import sys
from pathlib import Path
import re
from datetime import datetime

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

def main():
    url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    # ==========================================
    # RealVisXL V5.0 + IP-Adapter設定
    # 参照画像の人物からフォトリアル画像を生成
    # ==========================================

    # 参照画像のパス（人物写真）
    reference_image_path = "taiwanese01.png"  # ここに参照画像のパスを指定

    # フォトリアリスティックプロンプト
    prompt_ja = """
Portrait of a man, confident expression, natural lighting, detailed facial features, 
realistic skin texture, professional photography, sharp focus, cinematic composition,
casual clothing, outdoor background, depth of field, 8k uhd
"""

    # RealVisXL V5.0用の最適化されたnegative prompt
    negative_prompt_base = """
(octane render, render, drawing, anime, bad photo, bad photography:1.3), 
(worst quality, low quality, blurry:1.2), (bad teeth, deformed teeth, deformed lips), 
(bad anatomy, bad proportions:1.1), (deformed iris, deformed pupils), 
(deformed eyes, bad eyes), (deformed face, ugly face, bad face), 
(deformed hands, bad hands, fused fingers), morbid, mutilated, mutation, disfigured,
watermark, text, signature, logo
"""
    
    print("\n使用モデル: RealVisXL V5.0")
    print("生成手法: 1024px生成 → リサイズ → Img2Img (Strength 0.3)")

    # プロンプトを英語に翻訳（日本語が含まれている場合）
    prompt_en = translate_to_english(prompt_ja)

    payload = {
        "input": {
            "prompt": prompt_en.strip(),
            "negative_prompt": negative_prompt_base.strip(),
            "steps": 28,
            "guidance_scale": 6.0,
            "seed": 42,
            "width": 1536,
            "height": 1536,
            "ip_adapter_scale": 0.6,
            "scheduler": "Euler a",
            # LoRAの設定（例）
            # "loras": [
            #     {"path": "username/repo-name", "name": "skin", "weight": 0.6},
            #     {"path": "username/taiwanese-lora", "name": "face", "weight": 0.8}
            # ],
            # "lora_scale": 1.0  # 全体の効き具合
        }
    }

    # 参照画像が存在する場合は追加
    if os.path.exists(reference_image_path):
        print(f"📸 参照画像を読み込み: {reference_image_path}")
        payload["input"]["reference_image"] = encode_image_to_base64(reference_image_path)
        print("✓ 参照画像をエンコード完了")
        print(f"   IP-Adapter影響度: {payload['input']['ip_adapter_scale']}")
    else:
        print("⚠️  参照画像が見つかりません。通常のtext-to-imageで生成します。")
        print(f"   参照画像を使う場合: {reference_image_path} に画像を配置してください。")

    print("\nリクエスト送信中...")
    start_time = time.time()

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=600)
        
        # レスポンスのステータスコードを確認
        if response.status_code != 200:
            print(f"❌ HTTPエラー: {response.status_code}")
            print(f"   レスポンス: {response.text[:500]}")
        else:
            # JSONとしてパース
            try:
                response_data = response.json()
            except ValueError as json_err:
                print(f"❌ JSON解析エラー: {json_err}")
                print(f"   レスポンス内容: {response.text[:500]}")
                raise
            
            print(f"完了！ かかった時間: {time.time() - start_time:.2f}秒")

            # RunPodのステータスを確認
            if not isinstance(response_data, dict):
                print(f"❌ レスポンスが辞書ではありません: {type(response_data)}")
                print(f"   内容: {response_data}")
            elif response_data.get('status') in ['IN_PROGRESS', 'IN_QUEUE']:
                # ジョブがまだ処理中またはキュー待ちの場合、ステータスをポーリング
                job_id = response_data.get('id')
                current_status = response_data.get('status')
                print(f"⏳ ジョブ{current_status}... (ID: {job_id})")
                print(f"   ステータスを確認しています...")
                
                status_url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/status/{job_id}"
                max_retries = 60  # 最大60回（約10分）
                retry_interval = 10  # 10秒ごと
                
                for attempt in range(max_retries):
                    time.sleep(retry_interval)
                    status_response = requests.get(status_url, headers=headers)
                    status_data = status_response.json()
                    
                    current_status = status_data.get('status')
                    print(f"   [{attempt + 1}/{max_retries}] ステータス: {current_status}")
                    
                    if current_status == 'COMPLETED':
                        response_data = status_data
                        print(f"✅ ジョブ完了！ 合計時間: {time.time() - start_time:.2f}秒")
                        break
                    elif current_status == 'FAILED':
                        print(f"❌ ジョブ失敗: {status_data.get('error', 'Unknown error')}")
                        sys.exit(1)
                    elif current_status not in ['IN_PROGRESS', 'IN_QUEUE']:
                        print(f"⚠️  予期しないステータス: {current_status}")
                        print(f"   レスポンス: {status_data}")
                        sys.exit(1)
                else:
                    print(f"❌ タイムアウト: {max_retries * retry_interval}秒以内に完了しませんでした")
                    sys.exit(1)
            
            # 結果を処理
            if 'output' in response_data:
                output = response_data['output']
                
                # outputが辞書であることを確認
                if not isinstance(output, dict):
                    print(f"⚠️  予期しない出力形式: {type(output)}")
                    print(f"   内容: {output}")
                    print(f"\n   レスポンス全体:")
                    import json
                    print(json.dumps(response_data, indent=2, ensure_ascii=False))
                elif 'error' in output:
                    print(f"❌ サーバーエラー: {output['error']}")
                elif 'image' in output:
                    img_base64 = output['image']
                    
                    # タイムスタンプ付きファイル名を生成
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    prefix = "animagine_ip" if "reference_image" in payload["input"] else "animagine"
                    output_filename = f"output_{prefix}_{timestamp}.png"
                    
                    # 画像保存
                    image = Image.open(BytesIO(base64.b64decode(img_base64)))
                    image.save(output_filename)
                    
                    print(f"\n✅ 画像保存完了: {output_filename}")
                    print(f"   プロンプト: {output.get('prompt', 'N/A')[:80]}...")
                    print(f"   サイズ: {output.get('width', 'N/A')}x{output.get('height', 'N/A')}")
                    print(f"   ステップ数: {output.get('steps', 'N/A')}")
                    
                    if "reference_image" in payload["input"]:
                        print(f"   参照画像使用: はい (影響度: {payload['input']['ip_adapter_scale']})")
                else:
                    print("⚠️  予期せぬレスポンス形式:")
                    print(f"   型: {type(output)}")
                    print(f"   内容: {output}")
            else:
                print("❌ 'output'キーが見つかりません:")
                print(f"   レスポンスキー: {response_data.keys() if isinstance(response_data, dict) else 'N/A'}")
                print(f"   内容: {response_data}")

    except requests.exceptions.Timeout:
        print(f"❌ タイムアウトエラー: 600秒以内に応答がありませんでした")
    except requests.exceptions.RequestException as req_err:
        print(f"❌ リクエストエラー: {req_err}")
    except Exception as e:
        print(f"❌ 予期せぬエラー: {type(e).__name__}: {e}")
        import traceback
        print("\n詳細なトレースバック:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
