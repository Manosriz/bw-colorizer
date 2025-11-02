import os
import io
import time
import urllib.request
from pathlib import Path

import numpy as np
import cv2 as cv
import streamlit as st

# -----------------------------
# Config & constants
# -----------------------------
STORAGE = Path("./models")
STORAGE.mkdir(parents=True, exist_ok=True)

URLS = {
    "prototxt": "https://storage.openvinotoolkit.org/repositories/datumaro/models/colorization/colorization_deploy_v2.prototxt",
    "caffemodel": "https://storage.openvinotoolkit.org/repositories/datumaro/models/colorization/colorization_release_v2.caffemodel",
    "pts_in_hull": "https://storage.openvinotoolkit.org/repositories/datumaro/models/colorization/pts_in_hull.npy",
}

FILES = {
    "prototxt": STORAGE / "colorization_deploy_v2.prototxt",
    "caffemodel": STORAGE / "colorization_release_v2.caffemodel",
    "pts_in_hull": STORAGE / "pts_in_hull.npy",
}

# -----------------------------
# Utilities
# -----------------------------
@st.cache_data(show_spinner=False)
def _download_once(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        f.write(r.read())

@st.cache_resource(show_spinner=True)
def load_model():
    # Ensure model assets exist
    for k in FILES:
        _download_once(URLS[k], FILES[k])

    net = cv.dnn.readNetFromCaffe(str(FILES["prototxt"]), str(FILES["caffemodel"]))
    pts = np.load(str(FILES["pts_in_hull"]))  # (313, 2)
    pts = pts.transpose().reshape(2, 313, 1, 1).astype(np.float32)

    # Inject cluster centers and rebalancing into the network
    class8_id = net.getLayerId("class8_ab")
    conv8_id = net.getLayerId("conv8_313_rh")

    net.setParam(class8_id, 0, pts)
    net.setParam(conv8_id, 0, np.full((1, 313), 2.606, dtype=np.float32))
    return net

def ensure_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Return an RGB image in [0,1]. If uploaded is grayscale, promote to 3ch."""
    if image_bgr.ndim == 2:
        image_bgr = cv.cvtColor(image_bgr, cv.COLOR_GRAY2BGR)
    elif image_bgr.shape[2] == 1:
        image_bgr = cv.cvtColor(image_bgr, cv.COLOR_GRAY2BGR)
    img_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img_rgb

def colorize(net, img_rgb: np.ndarray, resize_to: int = 224) -> np.ndarray:
    """
    Colorize with Zhang et al. model.
    img_rgb: float32 RGB in [0,1]
    returns RGB in [0,1]
    """
    H, W = img_rgb.shape[:2]
    img_lab = cv.cvtColor(img_rgb, cv.COLOR_RGB2LAB).astype(np.float32)
    L = img_lab[:, :, 0]

    # Network expects 224x224 and L shifted by -50
    L_rs = cv.resize(L, (resize_to, resize_to))
    L_rs -= 50

    net.setInput(cv.dnn.blobFromImage(L_rs))
    ab_dec = net.forward()[0, :, :, :].transpose((1, 2, 0))

    # Resize ab to original size
    ab_up = cv.resize(ab_dec, (W, H))
    lab_out = np.concatenate([L[..., np.newaxis], ab_up], axis=2).astype(np.float32)
    img_out = cv.cvtColor(lab_out, cv.COLOR_LAB2RGB)
    img_out = np.clip(img_out, 0, 1)
    return img_out

def read_image(file) -> np.ndarray:
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv.imdecode(file_bytes, cv.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not decode image.")
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv.cvtColor(img, cv.COLOR_BGRA2BGR)
    return img

def enhance_contrast_rgb(img_rgb: np.ndarray) -> np.ndarray:
    """Optional: mild LAB CLAHE on L channel"""
    lab = cv.cvtColor(img_rgb, cv.COLOR_RGB2LAB)
    L, a, b = cv.split(lab)

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L2 = clahe.apply((L * 255).astype(np.uint8)).astype(np.float32) / 255.0 * 100.0
    lab2 = cv.merge([L2, a.astype(np.float32), b.astype(np.float32)])

    rgb2 = cv.cvtColor(lab2, cv.COLOR_LAB2RGB)
    return np.clip(rgb2, 0, 1)

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="B&W Photo Colorizer", page_icon="🎨", layout="centered")

st.title("🎨 Black-&-White Photo Colorizer")
st.caption("ECCV’16 Colorization (OpenCV DNN) • CPU-friendly • Auto-download model")

with st.sidebar:
    st.header("Settings")
    apply_clahe = st.checkbox("Enhance contrast (CLAHE)", value=True)
    preview_width = st.slider("Preview width (px)", 400, 1200, 700, step=50)

st.write("Upload a **black & white** (grayscale/monochrome) photo (JPG/PNG/TIFF).")

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded:
    try:
        raw_bgr = read_image(uploaded)
        rgb = ensure_rgb(raw_bgr)

        st.subheader("Original")
        st.image(rgb, use_column_width=True, clamp=True)

        with st.spinner("Loading model & colorizing…"):
            net = load_model()
            start = time.time()
            colorized = colorize(net, rgb)
            if apply_clahe:
                colorized = enhance_contrast_rgb(colorized)
            elapsed = time.time() - start

        st.subheader("Colorized")
        st.image(colorized, width=preview_width, clamp=True)
        st.success(f"Done in {elapsed:.2f}s on CPU.")

        # Download button
        out_bgr = cv.cvtColor((colorized * 255).astype(np.uint8), cv.COLOR_RGB2BGR)
        ok, buf = cv.imencode(".jpg", out_bgr, [int(cv.IMWRITE_JPEG_QUALITY), 95])
        if ok:
            st.download_button(
                label="Download colorized image (JPG)",
                data=io.BytesIO(buf.tobytes()),
                file_name="colorized.jpg",
                mime="image/jpeg",
            )
    except Exception as e:
        st.error(f"Oops — {e}")

st.markdown(
    "<hr><small>Model: Zhang, Isola, Efros (ECCV 2016). Implemented via OpenCV DNN.</small>",
    unsafe_allow_html=True,
)
