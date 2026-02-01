import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, AutoencoderKL, DPMSolverMultistepScheduler
from diffusers.utils import load_image
import runpod
import io
import base64
import sys
import traceback
from PIL import Image

print("=" * 60)
print("RunPod Serverless Worker - Initialization Started")
print("=" * 60)

try:
    # デバイスの自動検出
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✓ Device: {device}")
    
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # モデルの初期化（グローバルスコープで1回だけ実行）
    print("\n[1/3] Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", 
        torch_dtype=torch.float16
    )
    print("✓ VAE loaded")
    
    print("\n[2/3] Loading SDXL pipeline (this may take a few minutes)...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "SG161222/RealVisXL_V4.0",
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    print("✓ Pipeline loaded (RealVisXL V4.0)")
    
    # GPUへ転送
    print("\n[3/3] Moving model to device...")
    pipe.to(device)
    print(f"✓ Model moved to {device}")
    
    # Img2Imgパイプラインも初期化
    print("\n[3.5/4] Loading Img2Img pipeline...")
    img2img_pipe = StableDiffusionXLImg2ImgPipeline(
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        text_encoder_2=pipe.text_encoder_2,
        tokenizer=pipe.tokenizer,
        tokenizer_2=pipe.tokenizer_2,
        unet=pipe.unet,
        scheduler=pipe.scheduler,
    )
    img2img_pipe.to(device)
    print("✓ Img2Img pipeline loaded")
    
    # 商用利用向け：見えない透かし（invisible-watermark）を無効化
    if hasattr(pipe, "watermark"):
        pipe.watermark = None
        print("✓ Watermark disabled")
    
    # IP-Adapterのロード（参照画像機能用）
    print("\n[4/4] Loading IP-Adapter...")
    try:
        pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="sdxl_models",
            weight_name="ip-adapter_sdxl.bin"
        )
        pipe.set_ip_adapter_scale(0.6)  # デフォルトの影響度
        print("✓ IP-Adapter loaded")
    except Exception as e:
        print(f"⚠️  IP-Adapter load failed (optional): {e}")
        print("   Text-to-image will still work")
    
    print("\n" + "=" * 60)
    print("✓ Model initialization completed successfully!")
    print("=" * 60 + "\n")
    
except Exception as e:
    print("\n" + "=" * 60)
    print("❌ ERROR during initialization:")
    print("=" * 60)
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    print("\nFull traceback:")
    traceback.print_exc()
    print("=" * 60)
    sys.exit(1)


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
        job_id = job.get("id", "unknown")
        
        print(f"\n{'='*60}")
        print(f"Processing Job: {job_id}")
        print(f"{'='*60}")
        
        # パラメータの取得（デフォルト値あり）
        prompt = job_input.get("prompt", "a simple landscape")
        negative_prompt = job_input.get("negative_prompt", "low quality, blurry")
        steps = job_input.get("steps", 30)
        cfg_scale = job_input.get("guidance_scale", 7.5)
        seed = job_input.get("seed", None)
        width = job_input.get("width", 1024)
        height = job_input.get("height", 1024)
        reference_image_b64 = job_input.get("reference_image", None)
        ip_adapter_scale = job_input.get("ip_adapter_scale", 0.6)
        scheduler_type = job_input.get("scheduler", "default")
        
        print(f"Prompt: {prompt[:100]}...")
        print(f"Size: {width}x{height}, Steps: {steps}, CFG: {cfg_scale}")
        print(f"Scheduler: {scheduler_type}")
        
        # スケジューラーの設定
        if scheduler_type == "DPM++ 2M Karras":
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config,
                use_karras_sigmas=True,
                algorithm_type="dpmsolver++"
            )
            img2img_pipe.scheduler = pipe.scheduler
            print("✓ Scheduler set to DPM++ 2M Karras")
        
        # バリデーション
        if not prompt:
            return {"error": "Input is missing the 'prompt' key. Please include a prompt."}
        
        # 参照画像の処理（IP-Adapter用）
        reference_image = None
        if reference_image_b64:
            try:
                print("Decoding reference image...")
                img_data = base64.b64decode(reference_image_b64)
                reference_image = Image.open(io.BytesIO(img_data)).convert("RGB")
                pipe.set_ip_adapter_scale(ip_adapter_scale)
                print(f"✓ Reference image loaded (IP-Adapter scale: {ip_adapter_scale})")
            except Exception as e:
                print(f"⚠️  Failed to decode reference image: {e}")
                reference_image = None
        
        # シード値の設定（再現性のため）
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
            print(f"Seed: {seed}")
        
        # 進捗更新
        mode = "IP-Adapter generation" if reference_image else "Text-to-image generation"
        runpod.serverless.progress_update(job, f"{mode}...")
        print(f"Starting {mode}...")
        
        # 画像生成実行: Step 1 - 1024pxで生成
        print("Step 1/3: Generating 1024px base image...")
        with torch.inference_mode():
            generation_kwargs = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_inference_steps": steps,
                "guidance_scale": cfg_scale,
                "width": 1024,
                "height": 1024,
                "generator": generator,
            }
            
            # 参照画像がある場合はIP-Adapterを使用
            if reference_image:
                generation_kwargs["ip_adapter_image"] = reference_image
            
            base_image = pipe(**generation_kwargs).images[0]
        
        print("✓ Base image generated at 1024x1024")
        
        # Step 2 - 目標サイズにリサイズ
        print(f"Step 2/3: Resizing to {width}x{height}...")
        resized_image = base_image.resize((width, height), Image.LANCZOS)
        print("✓ Image resized")
        
        # Step 3 - Img2Img (Strength 0.3)で品質向上
        print("Step 3/3: Applying Img2Img refinement (strength=0.3)...")
        with torch.inference_mode():
            img2img_kwargs = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "image": resized_image,
                "strength": 0.3,
                "num_inference_steps": steps,
                "guidance_scale": cfg_scale,
                "generator": generator,
            }
            
            image = img2img_pipe(**img2img_kwargs).images[0]
        
        print("✓ Image refined with Img2Img")
        
        # 画像をBase64に変換
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        img_size_mb = len(img_str) / 1024 / 1024
        
        print(f"✓ Image encoded (size: {img_size_mb:.2f} MB)")
        print(f"{'='*60}\n")
        
        # 結果を返す
        return {
            "image": img_str,
            "prompt": prompt,
            "seed": seed,
            "steps": steps,
            "width": width,
            "height": height
        }
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"\n❌ ERROR in handler:")
        print(error_msg)
        traceback.print_exc()
        return {"error": error_msg}


# RunPod Serverlessを起動
runpod.serverless.start({"handler": handler})
