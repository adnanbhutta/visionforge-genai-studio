import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io
import os
import tempfile

if hasattr(Image, "Resampling"):
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
else:
    RESAMPLE_LANCZOS = getattr(Image, "LANCZOS", 1)

# ─── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="Q2 — Pix2Pix Sketch to Image",
    page_icon="✏️",
    layout="wide"
)

# ─── Model Definition ────────────────────────────────────────
class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, down=True, use_bn=True, dropout=False):
        super().__init__()
        if down:
            self.conv = nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=not use_bn)
        else:
            self.conv = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=not use_bn)
        self.bn      = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.act     = nn.LeakyReLU(0.2) if down else nn.ReLU()
        self.dropout = nn.Dropout(0.5) if dropout else nn.Identity()

    def forward(self, x):
        return self.dropout(self.bn(self.act(self.conv(x))))


class UNetGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, f=64):
        super().__init__()
        self.e1 = nn.Sequential(nn.Conv2d(in_ch, f,   4,2,1), nn.LeakyReLU(0.2))
        self.e2 = UNetBlock(f,    f*2,  down=True)
        self.e3 = UNetBlock(f*2,  f*4,  down=True)
        self.e4 = UNetBlock(f*4,  f*8,  down=True)
        self.e5 = UNetBlock(f*8,  f*8,  down=True)
        self.e6 = UNetBlock(f*8,  f*8,  down=True)
        self.e7 = UNetBlock(f*8,  f*8,  down=True)
        self.e8 = UNetBlock(f*8,  f*8,  down=True, use_bn=False)

        self.d1 = UNetBlock(f*8,    f*8, down=False, dropout=True)
        self.d2 = UNetBlock(f*8*2,  f*8, down=False, dropout=True)
        self.d3 = UNetBlock(f*8*2,  f*8, down=False, dropout=True)
        self.d4 = UNetBlock(f*8*2,  f*8, down=False)
        self.d5 = UNetBlock(f*8*2,  f*4, down=False)
        self.d6 = UNetBlock(f*4*2,  f*2, down=False)
        self.d7 = UNetBlock(f*2*2,  f,   down=False)
        self.d8 = nn.Sequential(
            nn.ConvTranspose2d(f*2, out_ch, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.e1(x);  e2 = self.e2(e1); e3 = self.e3(e2); e4 = self.e4(e3)
        e5 = self.e5(e4); e6 = self.e6(e5); e7 = self.e7(e6); e8 = self.e8(e7)
        d1 = self.d1(e8)
        d2 = self.d2(torch.cat([d1, e7], 1))
        d3 = self.d3(torch.cat([d2, e6], 1))
        d4 = self.d4(torch.cat([d3, e5], 1))
        d5 = self.d5(torch.cat([d4, e4], 1))
        d6 = self.d6(torch.cat([d5, e3], 1))
        d7 = self.d7(torch.cat([d6, e2], 1))
        return self.d8(torch.cat([d7, e1], 1))


device = torch.device('cpu')
IMG_SIZE = 256


# ─── Helper Functions ────────────────────────────────────────
def load_generator(weights_path):
    model = UNetGenerator().to(device)
    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location=device)
        if isinstance(state, dict) and 'G_state_dict' in state:
            state = state['G_state_dict']
        elif isinstance(state, dict) and 'G' in state:
            state = state['G']
        new_state = {k.replace('module.', ''): v for k, v in state.items()}
        model.load_state_dict(new_state)
        model.eval()
        return model, True
    model.eval()
    return model, False


def preprocess_image(img: Image.Image, size=256):
    img = img.convert('RGB').resize((size, size), RESAMPLE_LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5   # normalize to [-1, 1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


def postprocess_tensor(tensor):
    img = tensor.squeeze(0).permute(1, 2, 0).numpy()
    img = (img * 0.5 + 0.5).clip(0, 1)
    return Image.fromarray((img * 255).astype(np.uint8))


def save_uploaded_file_with_progress(uploaded_file, destination_path, chunk_size=8 * 1024 * 1024):
    total_size = getattr(uploaded_file, "size", 0) or 0
    written = 0

    progress = st.progress(0)
    status = st.empty()

    uploaded_file.seek(0)
    with open(destination_path, 'wb') as f:
        while True:
            chunk = uploaded_file.read(chunk_size)
            if not chunk:
                break

            f.write(chunk)
            written += len(chunk)

            if total_size > 0:
                pct = min(int((written / total_size) * 100), 100)
                progress.progress(pct)
                status.caption(
                    f"Saving uploaded weights: {pct}% "
                    f"({written / (1024 * 1024):.1f}/{total_size / (1024 * 1024):.1f} MB)"
                )
            else:
                status.caption(f"Saving uploaded weights: {written / (1024 * 1024):.1f} MB")

    progress.progress(100)
    status.caption("Upload complete. Weights saved successfully.")
    uploaded_file.seek(0)


# ─── UI ──────────────────────────────────────────────────────
st.title("✏️ Q2 — Pix2Pix: Sketch → Realistic Image")
st.markdown("""
**Course:** Generative AI (AI4009) &nbsp;|&nbsp; **Assignment:** 03 &nbsp;|&nbsp; **FAST NUCES**

Upload a **sketch or grayscale image** and the Pix2Pix model will generate a realistic colored version.
Trained on Anime Sketch Colorization dataset.
""")

st.divider()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    img_size = st.selectbox("Output resolution", [256, 128], index=0)

    st.divider()
    st.subheader("📁 Model Weights")
    weights_path = st.text_input("Generator weights path",
                                  value="output/pix2pix_generator_final.pth")
    weights_upload = st.file_uploader("Or upload weights (.pth)", type=['pth'])

    if weights_upload:
        weights_path = os.path.join(tempfile.gettempdir(), "pix2pix_uploaded.pth")
        upload_sig = f"{weights_upload.name}:{weights_upload.size}"

        if st.session_state.get("pix2pix_upload_sig") != upload_sig or not os.path.exists(weights_path):
            save_uploaded_file_with_progress(weights_upload, weights_path)
            st.session_state["pix2pix_upload_sig"] = upload_sig
        else:
            st.caption("Using uploaded weights already saved in this session.")

# Load model
generator, loaded = load_generator(weights_path)

if loaded:
    st.success("✅ Pix2Pix model loaded successfully")
else:
    st.warning("⚠️ Using random weights — upload your trained .pth file for real results")

st.divider()

# Main interface
tab1, tab2 = st.tabs(["🖼️ Single Image", "📊 Batch Compare"])

with tab1:
    st.subheader("Upload a Sketch")
    uploaded = st.file_uploader("Upload sketch image (.png, .jpg)", type=['png','jpg','jpeg'])

    if uploaded:
        input_img = Image.open(uploaded)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Input Sketch**")
            st.image(input_img, use_column_width=True)

        with col2:
            st.markdown("**Generated Output**")
            with st.spinner("Running Pix2Pix..."):
                tensor = preprocess_image(input_img, img_size)
                with torch.no_grad():
                    output = generator(tensor)
                result_img = postprocess_tensor(output)
            st.image(result_img, use_column_width=True)

            buf = io.BytesIO()
            result_img.save(buf, format='PNG')
            st.download_button("💾 Download Result",
                               buf.getvalue(), "pix2pix_output.png", "image/png")

        with col3:
            st.markdown("**Side by Side**")
            combined = Image.new('RGB', (input_img.width * 2, input_img.height))
            inp_resized = input_img.convert('RGB').resize(result_img.size)
            combined.paste(inp_resized, (0, 0))
            combined.paste(result_img, (result_img.width, 0))
            st.image(combined, use_column_width=True)

with tab2:
    st.subheader("Upload Multiple Sketches")
    multi_upload = st.file_uploader("Upload multiple images",
                                     type=['png','jpg','jpeg'],
                                     accept_multiple_files=True)

    if multi_upload and st.button("🚀 Generate All", type="primary"):
        cols = st.columns(min(len(multi_upload), 4))
        zip_buffers = {}

        for i, file in enumerate(multi_upload[:8]):
            img    = Image.open(file)
            tensor = preprocess_image(img, img_size)
            with torch.no_grad():
                output = generator(tensor)
            result = postprocess_tensor(output)

            buf = io.BytesIO()
            result.save(buf, format='PNG')
            out_name = f"pix2pix_{os.path.splitext(file.name)[0]}_output.png"
            zip_buffers[out_name] = buf.getvalue()

            with cols[i % 4]:
                st.markdown(f"**{file.name}**")
                st.image(img,    caption="Input",  use_column_width=True)
                st.image(result, caption="Output", use_column_width=True)
                st.download_button(
                    label="💾 Download",
                    data=buf.getvalue(),
                    file_name=out_name,
                    mime="image/png",
                    key=f"dl_{i}"
                )

        if zip_buffers:
            import zipfile
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, 'w') as zf:
                for fname, data in zip_buffers.items():
                    zf.writestr(fname, data)
            st.divider()
            st.download_button(
                label="📦 Download All as ZIP",
                data=zip_buf.getvalue(),
                file_name="pix2pix_outputs.zip",
                mime="application/zip"
            )

st.divider()

with st.expander("📖 How does Pix2Pix work?"):
    st.markdown("""
    ### Pix2Pix Architecture

    Pix2Pix is a **Conditional GAN (cGAN)** — the generator is conditioned on the input sketch.

    #### U-Net Generator
    - **Encoder**: Downsamples sketch → extracts features (structure, edges)
    - **Decoder**: Upsamples features → generates colored image
    - **Skip connections**: Copy encoder features directly to decoder → preserves fine details

    #### PatchGAN Discriminator
    - Instead of classifying the whole image, it classifies **30×30 patches**
    - Each patch is judged real or fake independently
    - Focuses on **local texture quality** instead of global structure

    #### Loss Functions
    | Loss | Formula | Purpose |
    |------|---------|---------|
    | Adversarial | BCE(D(sketch, fake), 1) | Fool discriminator |
    | L1 Reconstruction | λ × L1(fake, real) | Stay close to ground truth |
    | Total | Adv + 100×L1 | Balance realism + accuracy |

    The L1 loss (λ=100) is key — it forces the output to be close to the real image,
    preventing the generator from being too creative.
    """)

st.caption("Built with Streamlit | Generative AI Assignment 03 | FAST NUCES Spring 2026")