
from pathlib import Path
import torch,os
from unet import UNet
import cv2,base64
import numpy as np 
import albumentations as A
from albumentations.pytorch import ToTensorV2
from huggingface_hub import hf_hub_download
import torch
from typing import List, Tuple, Union, Optional
import sys, os                               
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
from pathlib import Path



#  loading model

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")


#  loading from huggingface

model=UNet(in_channels=3,num_classes=1).to(DEVICE)

WEIGHTS = hf_hub_download(
    repo_id="TejasB5/unet_brain_tumor",   
    filename="unet_best_fp16.pth",            
    cache_dir="/data/hf-cache"            
)
model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
model.eval()

torch.set_grad_enabled(False)

## load Unet 


IMG_SIZE=256
PROB_THRESH=0.3

# Albumentations pipeline
val_tf = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(), ToTensorV2()
])


def _preprocess(bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    out = val_tf(image=rgb)
    ten = out["image"].unsqueeze(0).to(DEVICE)          # 1×3×H×W
    return ten

def _postprocess(prob: np.ndarray,) -> np.ndarray:
    mask = (prob > PROB_THRESH).astype(np.uint8)          # 0/1
    # print(mask.shape)
    return mask * 255


def segment_one(img: Union[str, np.ndarray]) -> np.ndarray:
    if isinstance(img, (str, Path)):
        bgr = cv2.imread(str(img))
        if bgr is None:
            raise ValueError(f"Could not read image: {img}")
    else:
        bgr = img
    x = _preprocess(bgr)
    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0,0].cpu().numpy()    ## 1,1,256,256
    return _postprocess(prob)

def segment_batch(bgr_list: List[np.ndarray]) -> List[np.ndarray]:

    if len(bgr_list) == 0:
        return []

    tens = []
    for bgr in bgr_list:
        rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        out  = val_tf(image=rgb)
        tens.append(out["image"])
    batch = torch.stack(tens).to(DEVICE)          # N×3×H×W

    with torch.no_grad():
        probs = torch.sigmoid(model(batch)).cpu().numpy()   # N×1×H×W

    masks = [_postprocess(prob[0]) for prob in probs]       # List[N×H×W]

    
    return masks
# inference.py  ─ change ONLY this helper
def encode_png(arr: np.ndarray) -> str:
    """Return base-64 PNG string from an RGB or gray NumPy array."""
    # OpenCV encoders expect BGR
    if arr.ndim == 3 and arr.shape[2] == 3:          # RGB image
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise ValueError("PNG encoding failed")
    return base64.b64encode(buf).decode("utf-8")





