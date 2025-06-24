import streamlit as st
import cv2,requests,base64,io
import numpy as np

API_URL="http://127.0.0.1:8000/segment"
st.set_page_config(page_title="BRAIN-TUMOR SEGMENTATION",layout="centered")


st.title("Brain-tumour MRI Segmentation")
st.write("Upload MRI Scan (PNG/JPG/JPEG) to get predicted tumor")
uploaded = st.file_uploader("Choose MRI slice", type=["png", "jpg", "jpeg","tif","tiff"])

def b64_to_np(b64:str)->np.ndarray:
  data=base64.b64decode(b64)
  img=cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
  return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

if uploaded is not None:
  file_bytes=uploaded.getvalue()
  np_bgr=cv2.imdecode(np.frombuffer(file_bytes,np.uint8),cv2.IMREAD_COLOR)
  np_rgb=cv2.cvtColor(np_bgr, cv2.COLOR_BGR2RGB)
  st.image(np_rgb, caption="Original", width=300)

  if st.button("Segment"):
    with st.spinner("Running Model ......."):
      response=requests.post(API_URL,files={
        "file":(uploaded.name,file_bytes,uploaded.type)
      })
      response.raise_for_status()
    data=response.json()
    mask_rgb=b64_to_np(data["mask_b64"])
    # print(mask_rgb.shape,np_rgb.shape)


    mask = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2GRAY)  ## single channel grayscale image
    tumour_present = bool(np.any(mask > 0))
    overlap_rgb=np_rgb.copy()
    overlap_rgb[mask>0]=[255,0,0]

    if tumour_present:
        st.error("🩺  Tumour detected.")
    else:
        st.success("✅  No tumour detected.")


    st.subheader("Results")
    col1, col2 = st.columns(2)
    col1.image(mask_rgb,    caption="Predicted mask", width=300)
    col2.image(overlap_rgb, caption="Overlay",        width=300)


else:
  st.info("Upload an image to begin.")

   

