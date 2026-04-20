import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io
import os

# ─── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="Q1 — DCGAN vs WGAN-GP",
    page_icon="🎨",
    layout="wide"
)

# ─── Model Definitions (must match training code) ────────────
Z_DIM      = 100
CHANNELS   = 3
FEATURES_G = 64

class DCGANGenerator(nn.Module):
    def __init__(self, z=Z_DIM, ch=CHANNELS, f=FEATURES_G):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z,   f*8, 4,1,0,bias=False), nn.BatchNorm2d(f*8), nn.ReLU(True),
            nn.ConvTranspose2d(f*8, f*4, 4,2,1,bias=False), nn.BatchNorm2d(f*4), nn.ReLU(True),
            nn.ConvTranspose2d(f*4, f*2, 4,2,1,bias=False), nn.BatchNorm2d(f*2), nn.ReLU(True),
            nn.ConvTranspose2d(f*2, f,   4,2,1,bias=False), nn.BatchNorm2d(f),   nn.ReLU(True),
            nn.ConvTranspose2d(f,   ch,  4,2,1,bias=False), nn.Tanh()
        )
    def forward(self, z): return self.net(z)


class WGANGenerator(nn.Module):
    def __init__(self, z=Z_DIM, ch=CHANNELS, f=FEATURES_G):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z,   f*8, 4,1,0,bias=False), nn.BatchNorm2d(f*8), nn.ReLU(True),
            nn.ConvTranspose2d(f*8, f*4, 4,2,1,bias=False), nn.BatchNorm2d(f*4), nn.ReLU(True),
            nn.ConvTranspose2d(f*4, f*2, 4,2,1,bias=False), nn.BatchNorm2d(f*2), nn.ReLU(True),
            nn.ConvTranspose2d(f*2, f,   4,2,1,bias=False), nn.BatchNorm2d(f),   nn.ReLU(True),
            nn.ConvTranspose2d(f,   ch,  4,2,1,bias=False), nn.Tanh()
        )
    def forward(self, z): return self.net(z)


device = torch.device('cpu')

# ─── Helper Functions ────────────────────────────────────────
def load_model(model_class, weights_path):
    model = model_class().to(device)
    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location=device)
        if isinstance(state, dict) and 'G' in state:
            state = state['G']
        # Strip DataParallel prefix if present
        new_state = {}
        for k, v in state.items():
            new_state[k.replace('module.', '')] = v
        model.load_state_dict(new_state)
        model.eval()
        return model, True
    return model, False


def generate_images(model, n=16, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    with torch.no_grad():
        noise = torch.randn(n, Z_DIM, 1, 1, device=device)
        imgs  = model(noise).cpu()
    # Denormalize [-1,1] → [0,1]
    imgs = (imgs * 0.5 + 0.5).clamp(0, 1)
    return imgs


def tensor_to_grid(imgs, nrow=4):
    """Convert batch of tensors to a single grid PIL image."""
    n    = imgs.shape[0]
    nrow = min(nrow, n)
    ncol = (n + nrow - 1) // nrow
    _, c, h, w = imgs.shape
    grid = Image.new('RGB', (nrow * w, ncol * h), (255, 255, 255))
    for i, img in enumerate(imgs):
        pil = Image.fromarray((img.permute(1,2,0).numpy() * 255).astype(np.uint8))
        row, col = divmod(i, nrow)
        grid.paste(pil, (col * w, row * h))
    return grid


# ─── UI ──────────────────────────────────────────────────────
st.title("🎨 Q1 — DCGAN vs WGAN-GP: Tackling Mode Collapse")
st.markdown("""
**Course:** Generative AI (AI4009) &nbsp;|&nbsp; **Assignment:** 03 &nbsp;|&nbsp; **FAST NUCES**

This app demonstrates how **WGAN-GP** solves the mode collapse problem present in standard **DCGAN**.
Both models were trained on the Anime Faces dataset (21,551 images) on Kaggle T4×2 GPU.
""")

st.divider()

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Controls")
    n_images  = st.slider("Number of images to generate", 4, 16, 8, step=4)
    seed      = st.number_input("Random seed (0 = random)", min_value=0, value=42)
    use_seed  = seed != 0

    st.divider()
    st.subheader("📁 Model Weights")
    dcgan_path = st.text_input("DCGAN weights path",
                               value="outputs/dcgan_generator_final.pth")
    wgan_path  = st.text_input("WGAN-GP weights path",
                               value="outputs/wgan_generator_final.pth")
    st.info("Upload your trained .pth files below if paths don't exist.")

    dcgan_upload = st.file_uploader("Upload DCGAN weights (.pth)", type=['pth'])
    wgan_upload  = st.file_uploader("Upload WGAN-GP weights (.pth)", type=['pth'])

# Handle uploaded weights
if dcgan_upload:
    dcgan_path = "/tmp/dcgan_uploaded.pth"
    with open(dcgan_path, 'wb') as f:
        f.write(dcgan_upload.read())

if wgan_upload:
    wgan_path = "/tmp/wgan_uploaded.pth"
    with open(wgan_path, 'wb') as f:
        f.write(wgan_upload.read())

# Load models
dcgan_model, dcgan_loaded = load_model(DCGANGenerator, dcgan_path)
wgan_model,  wgan_loaded  = load_model(WGANGenerator,  wgan_path)

# Status indicators
col1, col2 = st.columns(2)
with col1:
    if dcgan_loaded:
        st.success("✅ DCGAN weights loaded")
    else:
        st.warning("⚠️ DCGAN using random weights (upload trained .pth)")
with col2:
    if wgan_loaded:
        st.success("✅ WGAN-GP weights loaded")
    else:
        st.warning("⚠️ WGAN-GP using random weights (upload trained .pth)")

st.divider()

# Generate button
if st.button("🚀 Generate Images", type="primary", use_container_width=True):
    seed_val = seed if use_seed else None

    col_d, col_w = st.columns(2)

    with col_d:
        st.subheader("🔵 DCGAN Generated Images")
        with st.spinner("Generating..."):
            dcgan_imgs = generate_images(dcgan_model, n_images, seed_val)
            dcgan_grid = tensor_to_grid(dcgan_imgs, nrow=4)
        st.image(dcgan_grid, width="stretch")
        buf = io.BytesIO()
        dcgan_grid.save(buf, format='PNG')
        st.download_button("💾 Download DCGAN Grid",
                           buf.getvalue(), "dcgan_generated.png", "image/png")

    with col_w:
        st.subheader("🟢 WGAN-GP Generated Images")
        with st.spinner("Generating..."):
            wgan_imgs = generate_images(wgan_model, n_images, seed_val)
            wgan_grid = tensor_to_grid(wgan_imgs, nrow=4)
        st.image(wgan_grid, width="stretch")
        buf2 = io.BytesIO()
        wgan_grid.save(buf2, format='PNG')
        st.download_button("💾 Download WGAN-GP Grid",
                           buf2.getvalue(), "wgan_generated.png", "image/png")

st.divider()

# Theory section
with st.expander("📖 What is Mode Collapse and how does WGAN-GP fix it?"):
    st.markdown("""
    ### Mode Collapse in GANs
    Mode collapse happens when the Generator learns to produce only a few types of images
    instead of the full variety of the training data. For example, instead of generating
    diverse anime faces, it keeps generating the same face over and over.

    ### Why DCGAN suffers from it
    - Uses **Binary Cross Entropy (BCE)** loss
    - Discriminator can become too strong → Generator gets no useful gradient
    - Training becomes unstable → Generator collapses to safe outputs

    ### How WGAN-GP fixes it
    - Replaces Discriminator with a **Critic** (no Sigmoid output)
    - Uses **Wasserstein Distance** instead of BCE — measures how different two distributions are
    - Adds **Gradient Penalty (λ=10)** — forces the Critic to be a 1-Lipschitz function
    - Result: **stable gradients always**, generator never collapses

    ### Key differences
    | Feature | DCGAN | WGAN-GP |
    |---------|-------|---------|
    | Loss function | BCE | Wasserstein |
    | Output activation | Sigmoid | None (raw score) |
    | Normalization | BatchNorm | InstanceNorm |
    | Critic updates | 1 per G update | 3-5 per G update |
    | Training stability | Can collapse | Stable |
    """)

st.caption("Built with Streamlit | Generative AI Assignment 03 | FAST NUCES Spring 2026")
