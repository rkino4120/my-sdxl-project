import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL

# Banana(Potassium)のフレームワークを使う場合
from potassium import Potassium, Request, Response

app = Potassium("my_app")

# 1. モデルの初期化（サーバー起動時に1回だけ実行される）
@app.init
def init():
    # VAE（色の補正などを行うパーツ）を軽量化して読み込み
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", 
        torch_dtype=torch.float16
    )
    
    # SDXL Baseモデルの読み込み
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    
    # GPUへ転送
    pipe.to("cuda")
    
    # 【重要】商用利用向け：見えない透かし（invisible-watermark）を無効化
    # これにより純粋な画像データのみが生成されます
    if hasattr(pipe, "watermark"):
        pipe.watermark = None

    context = {
        "pipe": pipe
    }
    return context

# 2. 生成リクエストの処理（APIを叩くたびに実行される）
@app.handler()
def handler(context: dict, request: Request) -> Response:
    pipe = context.get("pipe")
    
    # クライアントから送られてくるパラメータを受け取る
    prompt = request.json.get("prompt", "a simple landscape")
    negative_prompt = request.json.get("negative_prompt", "low quality")
    steps = request.json.get("steps", 30)
    cfg_scale = request.json.get("guidance_scale", 7.5)
    seed = request.json.get("seed", None)
    
    # シード値の固定（再現性のため）
    generator = None
    if seed is not None:
        generator = torch.Generator("cuda").manual_seed(seed)

    # 画像生成実行
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=cfg_scale,
        generator=generator,
    ).images[0]

    # 画像をBase64（文字列）に変換してクライアントに返す
    import io
    import base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return Response(
        json={"output": img_str}, 
        status=200
    )

if __name__ == "__main__":
    app.serve()