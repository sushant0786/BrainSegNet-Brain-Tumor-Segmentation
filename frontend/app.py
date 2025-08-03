
import streamlit as st

import requests, base64, cv2, numpy as np
from typing import List, Dict, Tuple
import os

API_URL = os.getenv("BACKEND_URL", st.secrets["backend_url"])



def b64_to_np(b64: str) -> np.ndarray:
    """base-64 PNG/JPG → RGB numpy array"""
    data = base64.b64decode(b64)
    bgr  = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip('#')
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return (r, g, b)


def overlay_mask(img_rgb: np.ndarray,mask_rgb: np.ndarray,colour: Tuple[int, int, int]) -> Tuple[np.ndarray, bool]:
    """Colour tumour pixels & return overlay + tumour flag"""
    gray = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2GRAY)
    out  = img_rgb.copy()
    out[gray > 0] = colour
    tumour = bool(np.any(gray > 0))
    return out, tumour
    


st.set_page_config(
    page_title="🧠 Brain-Tumour Segmentation",
    page_icon="🧠",
    
    layout="wide",
)

st.markdown(
    """
    <style>
    #MainMenu, footer{visibility:hidden;}

    section[data-testid="stSidebar"] > div:first-child{
        background:rgba(30,30,47,.35);
        backdrop-filter:blur(6px);
        border-right:1px solid rgba(255,255,255,.07);
    }

    div.stButton>button{
        border:0;border-radius:6px;padding:.6rem 1.4rem;
        background:linear-gradient(90deg,#ff4b4b 0%,#f95c8b 100%);
        color:#fff;font-weight:600;transition:box-shadow .25s;
    }
    div.stButton>button:hover{box-shadow:0 0 12px #ff4b4bbb;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ────────────────────────────────────────────────────────────────

st.sidebar.title("🖼️  Upload slices")
files = st.sidebar.file_uploader(
    "PNG / JPG / JPEG / TIF (multi-select OK)",
    type=["png", "jpg", "jpeg", "tif", "tiff"],
    accept_multiple_files=True,
)
mask_colour = hex_to_rgb(st.sidebar.color_picker("Mask colour", "#FF0000"))
st.sidebar.markdown("---")
st.sidebar.caption("🚀 FastAPI × Streamlit demo")


st.markdown("## Brain-tumour MRI segmentation demo")
st.write(
    "Upload one or many axial MRI slices.  The backend returns tumour masks "
    "in a single batched request; the app visualises the results."
)


if not files:
    st.info("⬅️  Select at least one image on the sidebar.")
    st.stop()


cols = st.columns(min(5, len(files)))
for c, f in zip(cols, files):
    c.image(f, width=110, caption=f.name)


if not st.button("🩺  Run segmentation"):
    st.stop()


payload = [("files", (f.name, f.getvalue(), f.type)) for f in files]

#  Backend call
with st.spinner("Contacting model server …"):
    bar = st.progress(0, "Uploading …")
    resp = requests.post(API_URL, files=payload, timeout=180)
    bar.progress(60, "Processing …")
    resp.raise_for_status()
    bar.progress(100, "Done 🎉")
    bar.empty()

data = resp.json()


if "results" in data:                # batch response
    results: List[Dict] = data["results"]
else:                                # fallback – single image old API
    results = [{
        "filename": files[0].name,
        "mask_b64": data["mask_b64"]
    }]

# Map original images to filename → RGB numpy
orig_np: Dict[str, np.ndarray] = {}
for f in files:
    bgr = cv2.imdecode(np.frombuffer(f.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    orig_np[f.name] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


if len(results) == 1:
    res           = results[0]
    name          = res["filename"]
    mask          = b64_to_np(res["mask_b64"])
    orig          = orig_np[name]
    overlay, flag = overlay_mask(orig, mask, mask_colour)

    (st.error if flag else st.success)(
        "⚠️ Tumour detected!" if flag else "✅ No tumour detected."
    )

    tab1, tab2, tab3 = st.tabs(["Mask", "Overlay", "Details"])
    with tab1:
        st.image(mask, caption="Predicted mask", width=420)
    with tab2:
        st.image(overlay, caption="Overlay", width=420)
    with tab3:
        h, w = orig.shape[:2]
        tumour_px = int(np.count_nonzero(cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)))
        st.markdown(f"""
**File**: {name}  
**Resolution**: {w} × {h}  
**Tumour pixels**: {tumour_px:,}  
**Tumour present**: {flag}  
""")

else:
    for idx, res in enumerate(results, 1):
        name  = res["filename"]
        mask  = b64_to_np(res["mask_b64"])
        orig  = orig_np[name]
        over, flag = overlay_mask(orig, mask, mask_colour)
        tumour_px  = int(np.count_nonzero(cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)))

        with st.expander(f"{idx}. {name}", expanded=True):
            c1, c2, c3 = st.columns(3)
            c1.image(orig, caption="Original", width=240)
            c2.image(mask, caption="Mask", width=240)
            c3.image(over, caption="Overlay", width=240)
            (st.error if flag else st.success)(
                "⚠️ Tumour detected!" if flag else "✅ No tumour detected."
            )
            st.caption(f"Tumour pixels: {tumour_px:,}")


