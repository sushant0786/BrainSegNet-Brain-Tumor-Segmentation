from fastapi import FastAPI,UploadFile,File,HTTPException
from starlette.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from typing import List, Tuple
import cv2,base64
import sys, os                               
import uvicorn
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
SRC_DIR  = os.path.join(BASE_DIR, "model", "src")
sys.path.append(SRC_DIR)
from inference import segment_one,encode_png,segment_batch

app=FastAPI(
  title="Brain Tumor Unet API",
  version="1.0.0",
  description="Returns binary tumour mask and red overlay",
  debug=True
)

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],           
  allow_methods=["POST"],
  allow_headers=["*"],
)

ALLOWED_TYPES = {
    "image/png", "image/jpeg", "image/jpg", "image/tiff", "image/tif"
}

def bgr2b64(arr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise ValueError("PNG encoding failed")
    return base64.b64encode(buf).decode("utf-8")


@app.post("/segment")
async def segment(files: List[UploadFile] = File(...)):
    """
    Accept 1…N image files in **one** request.

    • If 1 file → legacy response: {"mask_b64": …}
    • If >1    → batch response : {"results":[{filename,mask_b64}, …]}
    """
    if not files:
        raise HTTPException(400, "No files received.")

    images, names = [], []
    for f in files:
        if f.content_type not in ALLOWED_TYPES:
            raise HTTPException(415, f"{f.filename}: unsupported file type.")

        raw = await f.read()
        bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise HTTPException(400, f"{f.filename}: could not decode image.")
        images.append(bgr)
        names.append(f.filename)

    try:
        if len(images) == 1:
            masks = [segment_one(images[0])]
        else:
            masks = segment_batch(images)
            
    except Exception as e:
        raise HTTPException(500, f"Model inference failed: {e}")



    if len(masks) == 1:                                    # single-image path
        return JSONResponse({"mask_b64": encode_png(masks[0])})

    results = [
        {"filename": fn, "mask_b64": encode_png(msk)}
        for fn, msk in zip(names, masks)
    ]
    return JSONResponse({"results": results})



@app.get("/")
def ping():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)