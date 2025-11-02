import io
import time
import shutil
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import cv2 as cv
import streamlit as st
import sys
from pathlib import Path
DEOLDIFY_PATH = Path(__file__).parent / "third_party" / "DeOldify"
if DEOLDIFY_PATH.exists():
    sys.path.append(str(DEOLDIFY_PATH))
# =========================================
# Page config
# =========================================
st.set_page_config(page_title="B&W Photo Colorizer (Pro)", page_icon="🎨", layout="centered")

# =========================================
# Paths & model URLs (OpenCV model)
# =========================================
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

# =========================================
# Helpers
# =========================================
def _safe_rerun():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass

@st.cache_data(show_spinner=False)
def _download_once(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        f.write(r.read())

def reset_model_cache():
    """Delete downloaded model files so we can re-download clean copies."""
    if STORAGE.exists():
        shutil.rmtree(STORAGE, ignore_errors=True)
    STORAGE.mkdir(parents=True, exist_ok=True)

# =========================================
# Robust OpenCV model loader (classic)
# =========================================
@st.cache_resource(show_spinner=True)
def load_opencv_net():
    # Ensure assets exist
    for k in FILES:
        _download_once(URLS[k], FILES[k])

    # Load Caffe net
    net = cv.dnn.readNetFromCaffe(str(FILES["prototxt"]), str(FILES["caffemodel"]))

    # Load cluster centers (313x2) -> (2,313,1,1)
    pts = np.load(str(FILES["pts_in_hull"]))
    if pts.shape != (313, 2):
        raise ValueError(f"pts_in_hull.npy has unexpected shape {pts.shape}; try resetting model files.")
    pts = pts.transpose().reshape(2, 313, 1, 1).astype(np.float32)

    # Layer names per deploy_v2
    class8_name = "class8_ab"
    conv8rh_name = "conv8_313_rh"
    class8_id = net.getLayerId(class8_name)
    conv8_id = net.getLayerId(conv8rh_name)

    if class8_id <= 0 or conv8_id <= 0:
        raise RuntimeError(
            f"Expected layers not found: '{class8_name}', '{conv8rh_name}'. "
            f"Prototxt/caffemodel mismatch or corrupt download."
        )

    rebal = np.full((1, 313), 2.606, dtype=np.float32)

    # Try setParam; fall back to direct blob assignment
    try:
        net.setParam(class8_id, 0, pts)
    except cv.error:
        layer = net.getLayer(class8_id)
        layer.blobs = [pts]

    try:
        net.setParam(conv8_id, 0, rebal)
    except cv.error:
        layer = net.getLayer(conv8_id)
        layer.blobs = [rebal]

    return net

# =========================================
# DeOldify (Pro) — lazy import, optional
# =========================================
def _try_load_deoldify(artistic: bool = True):
    """
    Returns (colorizer, backend_name, error_msg)
    If not available, colorizer is None and error_msg tells how to enable.
    """
    try:
        # Imports
        from deoldify.visualize import get_image_colorizer
        # Pick model variant: artistic (richer colors) or stable (safer skin tones)
        colorizer = get_image_colorizer(artistic=artistic)
        # Detect device (MPS on Apple Silicon, CUDA, or CPU) - deoldify handles this internally via fastai
        backend = "DeOldify (artistic)" if artistic else "DeOldify (stable)"
        return colorizer, backend, None
    except Exception as e:
        msg = (
            "DeOldify not available. To enable Pro mode, add these to requirements and reinstall:\n\n"
            "  torch>=2.2\n  torchvision>=0.17\n  fastai==1.0.61\n  deoldify==0.6.0\n  Pillow>=9.5\n\n"
            "Then restart the app. (Large weights will auto-download on first use.)\n"
            f"Details: {e}"
        )
        return None, None, msg

def deoldify_colorize_rgb(np_img_rgb: np.ndarray, render_factor: int = 35, artistic: bool = True) -> np.ndarray:
    """
    Use DeOldify to colorize. Expects RGB float [0,1], returns RGB float [0,1].
    Implementation uses a temp file because DeOldify APIs are file-based.
    """
    # Lazy import to keep classic mode fast if Pro isn't used
    from PIL import Image
    from deoldify.visualize import get_image_colorizer

    colorizer = get_image_colorizer(artistic=artistic)

    # Save temp input as JPEG
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp_in:
        img_uint8 = (np.clip(np_img_rgb, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img_uint8).save(tmp_in.name, quality=95)
        # Get PIL image out
        pil_out = colorizer.get_transformed_image(tmp_in.name, render_factor=render_factor, watermarked=False)

    out = np.asarray(pil_out).astype(np.float32) / 255.0
    if out.ndim == 2:  # if grayscale somehow
        out = np.stack([out, out, out], axis=-1)
    return out

# =========================================
# Image utilities
# =========================================
def read_image(file) -> np.ndarray:
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv.imdecode(file_bytes, cv.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not decode image.")
    # If RGBA, drop alpha
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv.cvtColor(img, cv.COLOR_BGRA2BGR)
    return img

def ensure_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Return RGB float32 in [0,1]. If grayscale, promote to 3ch."""
    if image_bgr.ndim == 2:
        image_bgr = cv.cvtColor(image_bgr, cv.COLOR_GRAY2BGR)
    elif image_bgr.shape[2] == 1:
        image_bgr = cv.cvtColor(image_bgr, cv.COLOR_GRAY2BGR)
    img_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img_rgb

def colorize_opencv(net, img_rgb: np.ndarray, resize_to: int = 224) -> np.ndarray:
    """
    Classic model (Zhang et al.). img_rgb: float32 RGB in [0,1] -> returns RGB in [0,1]
    """
    H, W = img_rgb.shape[:2]
    img_lab = cv.cvtColor(img_rgb, cv.COLOR_RGB2LAB).astype(np.float32)
    L = img_lab[:, :, 0]  # [0,100]

    # Network expects 224x224 and L shifted by -50
    L_rs = cv.resize(L, (resize_to, resize_to))
    L_rs -= 50

    net.setInput(cv.dnn.blobFromImage(L_rs))
    ab_dec = net.forward()[0].transpose(1, 2, 0)  # (224,224,2)

    # Resize ab to original size
    ab_up = cv.resize(ab_dec, (W, H))
    lab_out = np.concatenate([L[..., np.newaxis], ab_up], axis=2).astype(np.float32)
    img_out = cv.cvtColor(lab_out, cv.COLOR_LAB2RGB)
    img_out = np.clip(img_out, 0, 1)
    return img_out

# ===== Quality Boost (no new deps) ===========================================
def boost_colors(img_rgb: np.ndarray, saturation_boost: float = 1.15, vibrance: float = 0.25, sharpen: bool = True) -> np.ndarray:
    """
    Simple 'vibrance'-style boost that increases low-sat regions more than high-sat.
    Also applies mild unsharp mask for details.
    """
    # Convert to HSV
    hsv = cv.cvtColor((img_rgb * 255).astype(np.uint8), cv.COLOR_RGB2HSV).astype(np.float32)
    H, S, V = cv.split(hsv)

    # Vibrance: boost more where saturation is low
    # S' = S * (1 + vibrance * (1 - S/255))
    S = S * (1.0 + vibrance * (1.0 - S / 255.0))
    S = np.clip(S, 0, 255)

    # Global saturation lift
    S *= saturation_boost
    S = np.clip(S, 0, 255)

    hsv2 = cv.merge([H, S, V]).astype(np.uint8)
    boosted = cv.cvtColor(hsv2, cv.COLOR_HSV2RGB).astype(np.float32) / 255.0

    if sharpen:
        # Mild unsharp mask
        blur = cv.GaussianBlur(boosted, (0, 0), sigmaX=1.0, sigmaY=1.0)
        boosted = np.clip(boosted * 1.0 + (boosted - blur) * 0.7, 0, 1)

    return boosted

def enhance_contrast_rgb(img_rgb: np.ndarray) -> np.ndarray:
    """CLAHE on L channel (LAB) to improve vintage photos."""
    lab = cv.cvtColor((img_rgb).astype(np.float32), cv.COLOR_RGB2LAB)
    L, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L2 = clahe.apply((np.clip(L,0,100) / 100.0 * 255).astype(np.uint8)).astype(np.float32) / 255.0 * 100.0
    lab2 = cv.merge([L2, a.astype(np.float32), b.astype(np.float32)])
    rgb2 = cv.cvtColor(lab2, cv.COLOR_LAB2RGB)
    return np.clip(rgb2, 0, 1)

# =========================================
# Sidebar
# =========================================
with st.sidebar:
    st.header("Settings")
    mode = st.radio("Quality mode", ["Pro (DeOldify)", "Classic (OpenCV)"], index=0)
    render_factor = st.slider("Pro: Render factor", 12, 45, 35, help="Higher = richer colors, slower. (DeOldify)")
    artistic = st.selectbox("Pro: Model variant", ["Artistic (richer)", "Stable (safer skin tones)"], index=0)
    apply_clahe = st.checkbox("Post: Enhance contrast (CLAHE)", value=True)
    vibrance = st.slider("Post: Vibrance", 0.0, 0.6, 0.25, 0.01)
    saturation_boost = st.slider("Post: Saturation boost", 1.0, 1.6, 1.15, 0.01)
    sharpen = st.checkbox("Post: Sharpen details", value=True)
    preview_width = st.slider("Preview width (px)", 400, 1200, 900, step=50)
    side_by_side = st.checkbox("Show side-by-side preview", value=True)
    if st.button("Reset model files (classic)"):
        reset_model_cache()
        st.cache_resource.clear()
        _safe_rerun()

# =========================================
# UI
# =========================================
st.title("🎨 Black-&-White Photo Colorizer — Upgraded")
st.caption("Pro mode uses DeOldify (PyTorch). Classic mode uses Zhang et al. (OpenCV DNN). Post-processing adds vibrance & detail.")

st.write("Upload a **black & white** (grayscale/monochrome) photo (JPG/PNG/TIFF).")
uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded:
    try:
        raw_bgr = read_image(uploaded)
        rgb = ensure_rgb(raw_bgr)

        if side_by_side:
            c1, c2 = st.columns(2, vertical_alignment="center")
            with c1:
                st.subheader("Original")
                st.image(rgb, use_column_width=True, clamp=True)
        else:
            st.subheader("Original")
            st.image(rgb, use_column_width=True, clamp=True)

        with st.spinner("Colorizing…"):
            t0 = time.time()
            if mode.startswith("Pro"):
                # Try DeOldify
                artistic_flag = artistic.startswith("Artistic")
                try:
                    out = deoldify_colorize_rgb(rgb, render_factor=render_factor, artistic=artistic_flag)
                except Exception as e:
                    # If DeOldify fails mid-run, show hint and fall back to classic
                    st.warning(f"Pro mode unavailable right now ({e}). Falling back to Classic.")
                    net = load_opencv_net()
                    out = colorize_opencv(net, rgb)
            else:
                net = load_opencv_net()
                out = colorize_opencv(net, rgb)

            # Post-processing (works for both modes)
            if apply_clahe:
                out = enhance_contrast_rgb(out)
            out = boost_colors(out, saturation_boost=saturation_boost, vibrance=vibrance, sharpen=sharpen)

            elapsed = time.time() - t0

        if side_by_side:
            with c2:
                st.subheader("Colorized")
                st.image(out, use_column_width=True, clamp=True)
        else:
            st.subheader("Colorized")
            st.image(out, width=preview_width, clamp=True)

        st.success(f"Done in {elapsed:.2f}s.")

        # Download button
        out_bgr = cv.cvtColor((out * 255).astype(np.uint8), cv.COLOR_RGB2BGR)
        ok, buf = cv.imencode(".jpg", out_bgr, [int(cv.IMWRITE_JPEG_QUALITY), 95])
        if ok:
            st.download_button(
                label="Download colorized image (JPG)",
                data=io.BytesIO(buf.tobytes()),
                file_name="colorized_pro.jpg",
                mime="image/jpeg",
            )

    except Exception as e:
        st.error(f"Oops — {e}")
        if mode.startswith("Pro"):
            _, _, hint = _try_load_deoldify()
            if hint:
                st.info(hint)

st.markdown(
    "<hr><small>Classic model: Zhang, Isola, Efros (ECCV 2016) via OpenCV DNN. "
    "Pro: DeOldify. Post-processing: vibrance, saturation, sharpen, CLAHE.</small>",
    unsafe_allow_html=True,
)
