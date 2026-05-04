import io
import logging
from contextlib import asynccontextmanager

import cv2
import numpy as np
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image

from realesrgan import RealESRGANer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

upsampler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global upsampler
    logger.info("Real-ESRGAN 모델 로딩 중...")

    half = torch.cuda.is_available()
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path="weights/RealESRGAN_x4plus.pth",
        model=model,
        tile=400,
        tile_pad=10,
        pre_pad=0,
        half=half,
    )

    logger.info(f"Real-ESRGAN 모델 로딩 완료 (device={'cuda' if half else 'cpu'}, tile=400)")
    yield
    del upsampler


app = FastAPI(title="Real-ESRGAN Upscale Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": upsampler is not None}


@app.post("/upscale")
async def upscale(
    image: UploadFile = File(...),
    outscale: float = Form(default=4.0),
):
    if upsampler is None:
        raise HTTPException(status_code=503, detail="모델 로딩 중입니다.")

    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 파일 오류: {e}")

    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    orig_h, orig_w = img.shape[:2]
    logger.info(f"업스케일 시작: {orig_w}x{orig_h} → x{outscale}")

    try:
        output, _ = upsampler.enhance(img, outscale=outscale)
    except RuntimeError as e:
        logger.error(f"추론 오류: {e}")
        raise HTTPException(status_code=500, detail=f"추론 오류: {e}")

    out_h, out_w = output.shape[:2]
    logger.info(f"업스케일 완료: {out_w}x{out_h}")

    _, buf = cv2.imencode(".png", output)
    return Response(content=buf.tobytes(), media_type="image/png")
