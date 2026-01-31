import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
import runpod
import io
import base64

# デバイスの自動検出
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# モデルの初期化（グローバルスコープで1回だけ実行）
print("Loading VAE...")
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", 
    torch_dtype=torch.float16
)

print("Loading SDXL pipeline...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)

# GPUへ転送
pipe.to(device)

# 商用利用向け：見えない透かし（invisible-watermark）を無効化
if hasattr(pipe, "watermark"):
    pipe.watermark = None

print("Model loaded successfully!")


def handler(job):
    """
    RunPod Serverless ハンドラー関数
    
    入力形式:
    {
        "input": {
            "prompt": "a beautiful landscape",
            "negative_prompt": "low quality",
            "steps": 30,
            "guidance_scale": 7.5,
            "seed": 42
        }
    }
    """
    try:
        # ジョブ入力の取得
        job_input = job["input"]
        
        # パラメータの取得（デフォルト値あり）
        prompt = job_input.get("prompt", "a simple landscape")
        negative_prompt = job_input.get("negative_prompt", "low quality, blurry")
        steps = job_input.get("steps", 30)
        cfg_scale = job_input.get("guidance_scale", 7.5)
        seed = job_input.get("seed", None)
        
        # バリデーション
        if not prompt:
            return {"error": "Input is missing the 'prompt' key. Please include a prompt."}
        
        # シード値の設定（再現性のため）
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
        
        # 進捗更新
        runpod.serverless.progress_update(job, "Generating image...")
        
        # 画像生成実行
        with torch.inference_mode():
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                generator=generator,
            ).images[0]
        
        # 画像をBase64に変換
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # 結果を返す
        return {
            "image": img_str,
            "prompt": prompt,
            "seed": seed,
            "steps": steps
        }
        
    except Exception as e:
        return {"error": str(e)}


# RunPod Serverlessを起動
runpod.serverless.start({"handler": handler})
