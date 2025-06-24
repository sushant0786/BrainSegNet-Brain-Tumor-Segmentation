from fastapi import FastAPI,UploadFile,File,HTTPException
from starlette.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from typing import Tuple
import cv2,base64
import sys, os                               
import uvicorn
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
SRC_DIR  = os.path.join(BASE_DIR, "model", "src")
sys.path.append(SRC_DIR)
from inference import segment_one,encode_png

app=FastAPI(
  title="Brain Tumor Unet API",
  version="1.0.0",
  description="Returns binary tumour mask and red overlay"
)

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],           
  allow_methods=["POST"],
  allow_headers=["*"],
)


def bgr2b64(arr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise ValueError("PNG encoding failed")
    return base64.b64encode(buf).decode("utf-8")


@app.post("/segment")
async def segment(file:UploadFile = File(...)):
    if file.content_type not in ("image/png", "image/jpeg", "image/jpg", "image/tiff"):
      raise HTTPException(415, "Only PNG or JPG or JPEG or TIF files are accepted.")
    
    data=await file.read()
    bgr=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)

    if bgr is None:
      raise HTTPException(400, "Could not decode image. Is it valid PNG/JPEG?")
    
    ## run model
    try:
      mask=segment_one(bgr)
    except Exception as e:
       raise HTTPException(400,"Could not decode image")
    

    return JSONResponse(
       {
          "mask_b64":bgr2b64(mask),           
       }
    )


@app.get("/")
def ping():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)