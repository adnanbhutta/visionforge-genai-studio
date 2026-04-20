# app.py — CycleGAN Sketch ↔ Photo  |  streamlit run app.py
# Requirements: streamlit torch torchvision pillow numpy scikit-image

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="CycleGAN: Sketch ↔ Photo",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# MODEL DEFINITION  (must match notebook exactly)
# ============================================================
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch, ch, 3),
            nn.InstanceNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch, ch, 3),
            nn.InstanceNorm2d(ch),
        )
    def forward(self, x):
        return x + self.block(x)


class ResNetGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, ngf=64, n_blocks=6):
        super().__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, ngf, 7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
        ]
        for mult in [1, 2]:
            model += [
                nn.Conv2d(ngf*mult, ngf*mult*2, 3, stride=2, padding=1),
                nn.InstanceNorm2d(ngf*mult*2),
                nn.ReLU(True),
            ]
        for _ in range(n_blocks):
            model += [ResBlock(ngf*4)]
        for mult in [2, 1]:
            model += [
                nn.ConvTranspose2d(ngf*mult*2, ngf*mult, 3,
                                   stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(ngf*mult),
                nn.ReLU(True),
            ]
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_ch, 7),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# ============================================================
# LOAD MODELS  (cached — only runs once per session)
# ============================================================
@st.cache_resource
def load_models(weight_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        ckpt = torch.load(weight_path, map_location=device)
    except FileNotFoundError:
        return None, None, None, f"Weight file not found: {weight_path}"
    except Exception as e:
        return None, None, None, str(e)

    cfg        = ckpt.get("cfg", {})
    n_blocks   = cfg.get("N_RES_BLOCKS", 6)

    def strip(sd):
        return {k.replace("module.", ""): v for k, v in sd.items()}

    G_AB = ResNetGenerator(n_blocks=n_blocks).to(device)
    G_BA = ResNetGenerator(n_blocks=n_blocks).to(device)
    G_AB.load_state_dict(strip(ckpt["G_AB"]))
    G_BA.load_state_dict(strip(ckpt["G_BA"]))
    G_AB.eval()
    G_BA.eval()
    return G_AB, G_BA, device, None


# ============================================================
# IMAGE HELPERS
# ============================================================
def preprocess(pil_img: Image.Image, size: int = 128) -> torch.Tensor:
    img = pil_img.convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0   # [0,255]→[-1,1]
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)


def postprocess(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    arr = ((arr.clip(-1, 1) + 1) * 127.5).astype(np.uint8)
    return Image.fromarray(arr)


def to_bytes(pil_img: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def compute_metrics(img_a: np.ndarray, img_b: np.ndarray):
    """SSIM and PSNR between two [0,1] numpy HWC arrays."""
    s = ssim_metric(img_a, img_b, data_range=1.0, channel_axis=2)
    p = psnr_metric(img_a, img_b, data_range=1.0)
    return s, p


def tensor_to_np01(t: torch.Tensor) -> np.ndarray:
    arr = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return ((arr.clip(-1, 1) + 1) / 2.0)   # [0,1]


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.title("⚙️ Settings")
    weight_path = st.text_input(
        "Weight file path",
        value="cyclegan_weights.pt",
        help="Path to the .pt file exported from your Kaggle notebook (Step 15).",
    )
    direction = st.radio(
        "Translation direction",
        ["🖊️ Sketch → Photo  (G_AB)", "🖼️ Photo → Sketch  (G_BA)"],
        index=0,
    )
    show_metrics   = st.toggle("Show SSIM / PSNR",       value=True)
    show_cycle     = st.toggle("Show cycle reconstruction", value=True)
    img_size_label = st.select_slider(
        "Display size",
        options=["Small", "Medium", "Large"],
        value="Medium",
    )

    st.markdown("---")
    st.markdown(
        "**How to get the weights:**\n"
        "1. Run the Kaggle notebook to completion\n"
        "2. Download `cyclegan_weights.pt` from `/kaggle/working/`\n"
        "3. Place it beside `app.py` and enter its path above"
    )

display_width = {"Small": 200, "Medium": 300, "Large": 400}[img_size_label]

# ============================================================
# LOAD MODELS
# ============================================================
G_AB, G_BA, device, err = load_models(weight_path)

if err:
    st.error(f"❌ Could not load weights: {err}")
    st.info("Make sure `cyclegan_weights.pt` is in the same folder as `app.py`, "
            "or enter the correct path in the sidebar.")
    st.stop()

st.sidebar.success(f"✅ Models loaded on **{device}**")

# ============================================================
# MAIN HEADER
# ============================================================
st.title("🎨 CycleGAN — Sketch ↔ Photo Translation")
st.caption(
    "Unpaired image-to-image translation using CycleGAN trained on the TU-Berlin Sketch dataset."
)

use_G_AB = "G_AB" in direction
generator = G_AB if use_G_AB else G_BA
inverse   = G_BA if use_G_AB else G_AB
input_label  = "Sketch" if use_G_AB else "Photo"
output_label = "Photo"  if use_G_AB else "Sketch"

# ============================================================
# UPLOAD
# ============================================================
st.subheader(f"📥 Upload a {input_label}")
uploaded = st.file_uploader(
    f"Choose a {input_label} image (PNG / JPG)",
    type=["png", "jpg", "jpeg", "bmp", "webp"],
)

if not uploaded:
    st.info("Upload an image above to get started.")
    st.stop()

# ============================================================
# INFERENCE
# ============================================================
pil_input = Image.open(uploaded).convert("RGB")

with torch.no_grad():
    t_in   = preprocess(pil_input).to(device)
    t_out  = generator(t_in)
    t_rec  = inverse(t_out)          # cycle reconstruction

pil_output = postprocess(t_out)
pil_rec    = postprocess(t_rec)

# numpy [0,1] arrays for metrics
np_in  = tensor_to_np01(t_in)
np_out = tensor_to_np01(t_out)
np_rec = tensor_to_np01(t_rec)

# ============================================================
# DISPLAY — 3 columns
# ============================================================
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"**Input {input_label}**")
    st.image(pil_input.resize((display_width, display_width), Image.BICUBIC),
             use_container_width=False)

with col2:
    st.markdown(f"**Generated {output_label}**")
    st.image(pil_output.resize((display_width, display_width), Image.BICUBIC),
             use_container_width=False)
    st.download_button(
        f"⬇️ Download {output_label}",
        data=to_bytes(pil_output),
        file_name=f"cyclegan_{output_label.lower()}.png",
        mime="image/png",
    )

with col3:
    if show_cycle:
        st.markdown(f"**Cycle Reconstruction ({input_label})**")
        st.image(pil_rec.resize((display_width, display_width), Image.BICUBIC),
                 use_container_width=False)
        st.download_button(
            "⬇️ Download Reconstruction",
            data=to_bytes(pil_rec),
            file_name="cyclegan_reconstructed.png",
            mime="image/png",
        )
    else:
        st.empty()

# ============================================================
# METRICS
# ============================================================
if show_metrics:
    st.markdown("---")
    st.subheader("📊 Quantitative Metrics")
    st.caption(
        "Computed between the **input** and its **cycle reconstruction** "
        "— measures how well structural information is preserved."
    )

    ssim_val, psnr_val = compute_metrics(np_in, np_rec)

    m1, m2, m3 = st.columns(3)
    m1.metric("SSIM  (↑ better, max 1.0)", f"{ssim_val:.4f}")
    m2.metric("PSNR  (↑ better, dB)",       f"{psnr_val:.2f}")
    m3.metric("Device",                      str(device).upper())

    # SSIM colour bar
    color = (
        "🟢 Excellent" if ssim_val > 0.80 else
        "🟡 Good"      if ssim_val > 0.65 else
        "🟠 Fair"      if ssim_val > 0.50 else
        "🔴 Poor"
    )
    st.markdown(f"**Cycle consistency quality:** {color}")

# ============================================================
# SIDE-BY-SIDE STRIP  (all 3 at fixed 128px — model resolution)
# ============================================================
st.markdown("---")
st.subheader("🔍 Model-Resolution Comparison  (128 × 128)")
st.caption("Images at the exact resolution the model operates on.")

strip_cols = st.columns(3)
labels_strip = [f"Input {input_label}", f"Translated {output_label}", "Reconstructed"]
tensors_strip = [t_in, t_out, t_rec]

for col, lbl, t in zip(strip_cols, labels_strip, tensors_strip):
    arr = (tensor_to_np01(t) * 255).astype(np.uint8)
    col.image(arr, caption=lbl, width=128)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    "<small>CycleGAN · TU-Berlin Sketch Dataset · "
    "ResNet Generator (6 blocks) · PatchGAN Discriminator · "
    "Trained with Mixed Precision on Kaggle T4 × 2</small>",
    unsafe_allow_html=True,
)