import io
import os
import tempfile
import zipfile

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image

try:
    from skimage.metrics import peak_signal_noise_ratio as psnr_metric
    from skimage.metrics import structural_similarity as ssim_metric

    SKIMAGE_AVAILABLE = True
except Exception:
    SKIMAGE_AVAILABLE = False

if hasattr(Image, "Resampling"):
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
else:
    RESAMPLE_LANCZOS = getattr(Image, "LANCZOS", 1)
    RESAMPLE_BICUBIC = getattr(Image, "BICUBIC", 3)


st.set_page_config(
    page_title="GenAI Assignment 03 - Combined App",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_modern_ui():
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at 10% 10%, #1f2a44 0%, #0b1120 40%, #04070f 100%);
            color: #e5e7eb;
        }
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #111827 60%, #0a0f1d 100%);
            border-right: 1px solid rgba(148, 163, 184, 0.22);
        }
        [data-testid="stSidebar"] * {
            color: #dbeafe !important;
        }
        div[data-testid="stMetric"] {
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(56, 189, 248, 0.30);
            border-radius: 14px;
            padding: 10px 14px;
        }
        .modern-hero {
            border: 1px solid rgba(56, 189, 248, 0.35);
            border-radius: 16px;
            padding: 18px 20px;
            margin-bottom: 1rem;
            background:
                linear-gradient(135deg, rgba(37, 99, 235, 0.18), rgba(14, 116, 144, 0.16)),
                rgba(15, 23, 42, 0.75);
            box-shadow: 0 12px 30px rgba(3, 7, 18, 0.35);
        }
        .modern-card {
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 14px;
            padding: 12px 14px;
            background: rgba(15, 23, 42, 0.64);
            margin-bottom: 0.8rem;
        }
        .modern-caption {
            color: #94a3b8;
            font-size: 0.95rem;
        }
        .stButton > button, .stDownloadButton > button {
            border-radius: 12px;
            border: 1px solid rgba(56, 189, 248, 0.45);
            background: linear-gradient(135deg, #2563eb, #0891b2);
            color: white;
            font-weight: 600;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            border-color: rgba(125, 211, 252, 0.85);
            box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.25);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_modern_ui()


Z_DIM = 100
FEATURES_G = 64
CHANNELS = 3
DEVICE = torch.device("cpu")


class DCGANGenerator(nn.Module):
    def __init__(self, z=Z_DIM, ch=CHANNELS, f=FEATURES_G):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z, f * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(f * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(f * 8, f * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(f * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(f * 4, f * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(f * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(f * 2, f, 4, 2, 1, bias=False),
            nn.BatchNorm2d(f),
            nn.ReLU(True),
            nn.ConvTranspose2d(f, ch, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class WGANGenerator(nn.Module):
    def __init__(self, z=Z_DIM, ch=CHANNELS, f=FEATURES_G):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z, f * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(f * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(f * 8, f * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(f * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(f * 4, f * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(f * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(f * 2, f, 4, 2, 1, bias=False),
            nn.BatchNorm2d(f),
            nn.ReLU(True),
            nn.ConvTranspose2d(f, ch, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, down=True, use_bn=True, dropout=False):
        super().__init__()
        if down:
            self.conv = nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=not use_bn)
        else:
            self.conv = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.act = nn.LeakyReLU(0.2) if down else nn.ReLU()
        self.dropout = nn.Dropout(0.5) if dropout else nn.Identity()

    def forward(self, x):
        return self.dropout(self.bn(self.act(self.conv(x))))


class UNetGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, f=64):
        super().__init__()
        self.e1 = nn.Sequential(nn.Conv2d(in_ch, f, 4, 2, 1), nn.LeakyReLU(0.2))
        self.e2 = UNetBlock(f, f * 2, down=True)
        self.e3 = UNetBlock(f * 2, f * 4, down=True)
        self.e4 = UNetBlock(f * 4, f * 8, down=True)
        self.e5 = UNetBlock(f * 8, f * 8, down=True)
        self.e6 = UNetBlock(f * 8, f * 8, down=True)
        self.e7 = UNetBlock(f * 8, f * 8, down=True)
        self.e8 = UNetBlock(f * 8, f * 8, down=True, use_bn=False)

        self.d1 = UNetBlock(f * 8, f * 8, down=False, dropout=True)
        self.d2 = UNetBlock(f * 8 * 2, f * 8, down=False, dropout=True)
        self.d3 = UNetBlock(f * 8 * 2, f * 8, down=False, dropout=True)
        self.d4 = UNetBlock(f * 8 * 2, f * 8, down=False)
        self.d5 = UNetBlock(f * 8 * 2, f * 4, down=False)
        self.d6 = UNetBlock(f * 4 * 2, f * 2, down=False)
        self.d7 = UNetBlock(f * 2 * 2, f, down=False)
        self.d8 = nn.Sequential(nn.ConvTranspose2d(f * 2, out_ch, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        e8 = self.e8(e7)
        d1 = self.d1(e8)
        d2 = self.d2(torch.cat([d1, e7], 1))
        d3 = self.d3(torch.cat([d2, e6], 1))
        d4 = self.d4(torch.cat([d3, e5], 1))
        d5 = self.d5(torch.cat([d4, e4], 1))
        d6 = self.d6(torch.cat([d5, e3], 1))
        d7 = self.d7(torch.cat([d6, e2], 1))
        return self.d8(torch.cat([d7, e1], 1))


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
                nn.Conv2d(ngf * mult, ngf * mult * 2, 3, stride=2, padding=1),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(True),
            ]
        for _ in range(n_blocks):
            model += [ResBlock(ngf * 4)]
        for mult in [2, 1]:
            model += [
                nn.ConvTranspose2d(
                    ngf * mult * 2,
                    ngf * mult,
                    3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.InstanceNorm2d(ngf * mult),
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


def strip_module_prefix(state_dict):
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


def load_model_state(model, weights_path, candidate_keys=None):
    model = model.to(DEVICE)
    model.eval()

    if not weights_path or not os.path.exists(weights_path):
        return model, False, f"File not found: {weights_path}"

    try:
        state = torch.load(weights_path, map_location=DEVICE)
        if candidate_keys and isinstance(state, dict):
            for key in candidate_keys:
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break

        if not isinstance(state, dict):
            raise ValueError("Checkpoint does not contain a valid state dict.")

        model.load_state_dict(strip_module_prefix(state))
        model.eval()
        return model, True, f"Loaded from {weights_path}"
    except Exception as exc:
        return model, False, f"Failed to load {weights_path}: {exc}"


def load_cyclegan_generators(weight_path):
    if not weight_path or not os.path.exists(weight_path):
        return None, None, False, f"File not found: {weight_path}"

    try:
        ckpt = torch.load(weight_path, map_location=DEVICE)
        if not isinstance(ckpt, dict) or "G_AB" not in ckpt or "G_BA" not in ckpt:
            raise ValueError("Checkpoint must contain 'G_AB' and 'G_BA'.")

        cfg = ckpt.get("cfg", {})
        n_blocks = cfg.get("N_RES_BLOCKS", 6) if isinstance(cfg, dict) else 6

        g_ab = ResNetGenerator(n_blocks=n_blocks).to(DEVICE)
        g_ba = ResNetGenerator(n_blocks=n_blocks).to(DEVICE)
        g_ab.load_state_dict(strip_module_prefix(ckpt["G_AB"]))
        g_ba.load_state_dict(strip_module_prefix(ckpt["G_BA"]))
        g_ab.eval()
        g_ba.eval()
        return g_ab, g_ba, True, f"Loaded from {weight_path}"
    except Exception as exc:
        return None, None, False, f"Failed to load {weight_path}: {exc}"


def resolve_uploaded_weights(uploaded_file, session_key, temp_name):
    if uploaded_file is None:
        return None

    temp_path = os.path.join(tempfile.gettempdir(), temp_name)
    signature = f"{uploaded_file.name}:{uploaded_file.size}"

    if st.session_state.get(session_key) != signature or not os.path.exists(temp_path):
        uploaded_file.seek(0)
        with open(temp_path, "wb") as handle:
            handle.write(uploaded_file.read())
        st.session_state[session_key] = signature
        uploaded_file.seek(0)

    return temp_path


def preprocess_image(img, size):
    img = img.convert("RGB").resize((size, size), RESAMPLE_LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


def postprocess_tensor(tensor):
    arr = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    arr = (arr * 0.5 + 0.5).clip(0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))


def generate_images(model, n=16, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    with torch.no_grad():
        noise = torch.randn(n, Z_DIM, 1, 1, device=DEVICE)
        imgs = model(noise).detach().cpu()
    return (imgs * 0.5 + 0.5).clamp(0, 1)


def tensor_to_grid(imgs, nrow=4):
    n = imgs.shape[0]
    nrow = min(nrow, n)
    ncol = (n + nrow - 1) // nrow
    _, _, h, w = imgs.shape
    grid = Image.new("RGB", (nrow * w, ncol * h), (255, 255, 255))

    for i, img in enumerate(imgs):
        pil = Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        row, col = divmod(i, nrow)
        grid.paste(pil, (col * w, row * h))

    return grid


def to_png_bytes(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def tensor_to_np01(tensor):
    arr = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    return (arr.clip(-1, 1) + 1) / 2.0


def compute_cycle_metrics(img_a, img_b):
    if not SKIMAGE_AVAILABLE:
        return None, None
    ssim_val = ssim_metric(img_a, img_b, data_range=1.0, channel_axis=2)
    psnr_val = psnr_metric(img_a, img_b, data_range=1.0)
    return ssim_val, psnr_val


st.sidebar.title("GenAI Studio")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    [
        "Home",
        "Q1 - DCGAN vs WGAN-GP",
        "Q2 - Pix2Pix",
        "Q3 - CycleGAN",
    ],
)

st.sidebar.divider()
st.sidebar.subheader("Model Weights")
st.sidebar.caption("Default paths are set for this workspace.")

dcgan_default = st.sidebar.text_input(
    "DCGAN generator path",
    value="output/dcgan_generator_final.pth",
)
wgan_default = st.sidebar.text_input(
    "WGAN-GP generator path",
    value="output/wgan_generator_final.pth",
)
pix_default = st.sidebar.text_input(
    "Pix2Pix generator path",
    value="output/pix2pix_generator_final.pth",
)
cyclegan_default = st.sidebar.text_input(
    "CycleGAN checkpoint path",
    value="cyclegan_weights.pt",
)

st.sidebar.caption("Optional: upload weights to override any path above.")
dcgan_upload = st.sidebar.file_uploader("Upload DCGAN (.pth)", type=["pth"], key="upload_dcgan")
wgan_upload = st.sidebar.file_uploader("Upload WGAN-GP (.pth)", type=["pth"], key="upload_wgan")
pix_upload = st.sidebar.file_uploader("Upload Pix2Pix (.pth)", type=["pth"], key="upload_pix")
cyc_upload = st.sidebar.file_uploader("Upload CycleGAN (.pt/.pth)", type=["pt", "pth"], key="upload_cyc")

dcgan_path = resolve_uploaded_weights(dcgan_upload, "sig_dcgan", "dcgan_uploaded.pth") or dcgan_default
wgan_path = resolve_uploaded_weights(wgan_upload, "sig_wgan", "wgan_uploaded.pth") or wgan_default
pix_path = resolve_uploaded_weights(pix_upload, "sig_pix", "pix2pix_uploaded.pth") or pix_default
cyclegan_path = (
    resolve_uploaded_weights(cyc_upload, "sig_cyc", "cyclegan_uploaded.pt") or cyclegan_default
)


if page == "Home":
    st.markdown(
        """
        <div class="modern-hero">
            <h2 style="margin:0;">Generative AI Studio</h2>
            <p class="modern-caption" style="margin:0.4rem 0 0 0;">
                One streamlined dashboard for Q1 (DCGAN vs WGAN-GP), Q2 (Pix2Pix), and Q3 (CycleGAN).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="modern-card">', unsafe_allow_html=True)
        st.subheader("Q1 - DCGAN vs WGAN-GP")
        st.caption("Random image generation from noise vectors.")
        if os.path.exists(dcgan_path) and os.path.exists(wgan_path):
            st.success("Q1 weights detected")
        else:
            st.warning("Q1 weights missing")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="modern-card">', unsafe_allow_html=True)
        st.subheader("Q2 - Pix2Pix")
        st.caption("Conditional image translation: sketch to image.")
        if os.path.exists(pix_path):
            st.success("Q2 weights detected")
        else:
            st.warning("Q2 weights missing")
        st.markdown("</div>", unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="modern-card">', unsafe_allow_html=True)
        st.subheader("Q3 - CycleGAN")
        st.caption("Unpaired translation with cycle consistency.")
        if os.path.exists(cyclegan_path):
            st.success("Q3 checkpoint detected")
        else:
            st.warning("Q3 checkpoint missing")
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown(
        """
        | Question | Model | Default Weight Path |
        |----------|-------|---------------------|
        | Q1 | DCGAN + WGAN-GP | output/*.pth |
        | Q2 | Pix2Pix (U-Net) | output/pix2pix_generator_final.pth |
        | Q3 | CycleGAN (ResNet) | cyclegan_weights.pt |
        """
    )


elif page == "Q1 - DCGAN vs WGAN-GP":
    st.title("Q1 - DCGAN vs WGAN-GP")
    st.markdown("Generate and compare image grids from both generators.")

    dcgan_model, dcgan_loaded, dcgan_msg = load_model_state(
        DCGANGenerator(), dcgan_path, candidate_keys=["G"]
    )
    wgan_model, wgan_loaded, wgan_msg = load_model_state(
        WGANGenerator(), wgan_path, candidate_keys=["G"]
    )

    s1, s2 = st.columns(2)
    with s1:
        if dcgan_loaded:
            st.success("DCGAN weights loaded")
        else:
            st.warning("DCGAN using random weights")
        st.caption(dcgan_msg)
    with s2:
        if wgan_loaded:
            st.success("WGAN-GP weights loaded")
        else:
            st.warning("WGAN-GP using random weights")
        st.caption(wgan_msg)

    st.divider()

    c1, c2, c3 = st.columns(3)
    with c1:
        n_images = st.slider("Number of images", 4, 16, 8, step=4)
    with c2:
        seed = st.number_input("Random seed (0 = random)", min_value=0, value=42)
    with c3:
        st.markdown("<br>", unsafe_allow_html=True)
        generate = st.button("Generate", type="primary", use_container_width=True)

    if generate:
        seed_val = None if seed == 0 else int(seed)
        left, right = st.columns(2)

        with left:
            st.subheader("DCGAN")
            with st.spinner("Generating..."):
                dcgan_imgs = generate_images(dcgan_model, n=n_images, seed=seed_val)
                dcgan_grid = tensor_to_grid(dcgan_imgs, nrow=4)
            st.image(dcgan_grid, use_container_width=True)
            st.download_button(
                "Download DCGAN Grid",
                data=to_png_bytes(dcgan_grid),
                file_name="dcgan_generated.png",
                mime="image/png",
            )

        with right:
            st.subheader("WGAN-GP")
            with st.spinner("Generating..."):
                wgan_imgs = generate_images(wgan_model, n=n_images, seed=seed_val)
                wgan_grid = tensor_to_grid(wgan_imgs, nrow=4)
            st.image(wgan_grid, use_container_width=True)
            st.download_button(
                "Download WGAN-GP Grid",
                data=to_png_bytes(wgan_grid),
                file_name="wgan_generated.png",
                mime="image/png",
            )

    st.divider()
    with st.expander("Q1 Notes"):
        st.markdown(
            """
            - DCGAN is prone to mode collapse when gradients become unstable.
            - WGAN-GP improves training stability using Wasserstein loss with gradient penalty.
            - Use the same seed to compare both models under identical noise input.
            """
        )


elif page == "Q2 - Pix2Pix":
    st.title("Q2 - Pix2Pix: Sketch to Image")
    st.markdown("Upload a sketch and generate a realistic output image.")

    generator, pix_loaded, pix_msg = load_model_state(
        UNetGenerator(), pix_path, candidate_keys=["G_state_dict", "G"]
    )

    if pix_loaded:
        st.success("Pix2Pix weights loaded")
    else:
        st.warning("Pix2Pix using random weights")
    st.caption(pix_msg)

    st.divider()

    img_size = st.selectbox("Output resolution", [256, 128], index=0)
    tab_single, tab_batch = st.tabs(["Single Image", "Batch Compare"])

    with tab_single:
        uploaded = st.file_uploader(
            "Upload sketch image (png/jpg/jpeg)",
            type=["png", "jpg", "jpeg"],
            key="q2_single_upload",
        )

        if uploaded:
            input_img = Image.open(uploaded)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Input Sketch**")
                st.image(input_img, use_container_width=True)

            with col2:
                st.markdown("**Generated Output**")
                with st.spinner("Running Pix2Pix..."):
                    tensor = preprocess_image(input_img, img_size)
                    with torch.no_grad():
                        output = generator(tensor)
                    result_img = postprocess_tensor(output)
                st.image(result_img, use_container_width=True)
                st.download_button(
                    "Download Result",
                    data=to_png_bytes(result_img),
                    file_name="pix2pix_output.png",
                    mime="image/png",
                )

            with col3:
                st.markdown("**Side by Side**")
                combined = Image.new("RGB", (result_img.width * 2, result_img.height))
                inp_resized = input_img.convert("RGB").resize(result_img.size, RESAMPLE_LANCZOS)
                combined.paste(inp_resized, (0, 0))
                combined.paste(result_img, (result_img.width, 0))
                st.image(combined, use_container_width=True)

    with tab_batch:
        multi_upload = st.file_uploader(
            "Upload multiple sketches",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="q2_multi_upload",
        )

        if multi_upload and st.button("Generate All", type="primary", key="q2_batch_generate"):
            cols = st.columns(min(len(multi_upload), 4))
            zip_buffers = {}

            for i, file in enumerate(multi_upload[:8]):
                img = Image.open(file)
                tensor = preprocess_image(img, img_size)
                with torch.no_grad():
                    output = generator(tensor)
                result = postprocess_tensor(output)

                out_name = f"pix2pix_{os.path.splitext(file.name)[0]}_output.png"
                png_data = to_png_bytes(result)
                zip_buffers[out_name] = png_data

                with cols[i % 4]:
                    st.markdown(f"**{file.name}**")
                    st.image(img, caption="Input", use_container_width=True)
                    st.image(result, caption="Output", use_container_width=True)
                    st.download_button(
                        label="Download",
                        data=png_data,
                        file_name=out_name,
                        mime="image/png",
                        key=f"q2_dl_{i}",
                    )

            if zip_buffers:
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, "w") as zf:
                    for fname, data in zip_buffers.items():
                        zf.writestr(fname, data)

                st.divider()
                st.download_button(
                    label="Download All as ZIP",
                    data=zip_buf.getvalue(),
                    file_name="pix2pix_outputs.zip",
                    mime="application/zip",
                )

    st.divider()
    with st.expander("Q2 Notes"):
        st.markdown(
            """
            - Pix2Pix is a conditional GAN that maps input sketch to target image.
            - U-Net skip connections help preserve edges and structure.
            - PatchGAN focuses on local realism in generated textures.
            """
        )


elif page == "Q3 - CycleGAN":
    st.title("Q3 - CycleGAN: Sketch and Photo Translation")
    st.markdown("Translate between domains with an unpaired CycleGAN checkpoint.")

    g_ab, g_ba, cyc_loaded, cyc_msg = load_cyclegan_generators(cyclegan_path)

    if cyc_loaded:
        st.success("CycleGAN checkpoint loaded")
    else:
        st.error("Could not load CycleGAN checkpoint")
    st.caption(cyc_msg)

    if not cyc_loaded:
        st.info("Set the correct CycleGAN checkpoint path or upload the checkpoint in the sidebar.")
        st.stop()

    assert g_ab is not None and g_ba is not None

    direction = st.radio(
        "Translation direction",
        ["Sketch -> Photo (G_AB)", "Photo -> Sketch (G_BA)"],
        index=0,
        horizontal=True,
    )
    show_metrics = st.toggle("Show SSIM and PSNR", value=True)
    show_cycle = st.toggle("Show cycle reconstruction", value=True)
    size_label = st.select_slider("Display size", options=["Small", "Medium", "Large"], value="Medium")
    display_width = {"Small": 200, "Medium": 300, "Large": 400}[size_label]

    uploaded = st.file_uploader(
        "Upload input image (png/jpg/jpeg/bmp/webp)",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
        key="q3_upload",
    )

    if uploaded:
        pil_input = Image.open(uploaded).convert("RGB")

        use_g_ab = "G_AB" in direction
        generator = g_ab if use_g_ab else g_ba
        inverse = g_ba if use_g_ab else g_ab
        input_label = "Sketch" if use_g_ab else "Photo"
        output_label = "Photo" if use_g_ab else "Sketch"

        with torch.no_grad():
            t_in = preprocess_image(pil_input, 128).to(DEVICE)
            t_out = generator(t_in)
            t_rec = inverse(t_out)

        pil_output = postprocess_tensor(t_out)
        pil_rec = postprocess_tensor(t_rec)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Input {input_label}**")
            st.image(
                pil_input.resize((display_width, display_width), RESAMPLE_BICUBIC),
                use_container_width=False,
            )

        with col2:
            st.markdown(f"**Generated {output_label}**")
            st.image(
                pil_output.resize((display_width, display_width), RESAMPLE_BICUBIC),
                use_container_width=False,
            )
            st.download_button(
                f"Download {output_label}",
                data=to_png_bytes(pil_output),
                file_name=f"cyclegan_{output_label.lower()}.png",
                mime="image/png",
            )

        with col3:
            if show_cycle:
                st.markdown(f"**Cycle Reconstruction ({input_label})**")
                st.image(
                    pil_rec.resize((display_width, display_width), RESAMPLE_BICUBIC),
                    use_container_width=False,
                )
                st.download_button(
                    "Download Reconstruction",
                    data=to_png_bytes(pil_rec),
                    file_name="cyclegan_reconstructed.png",
                    mime="image/png",
                )

        if show_metrics:
            st.divider()
            st.subheader("Metrics")

            if SKIMAGE_AVAILABLE:
                np_in = tensor_to_np01(t_in)
                np_rec = tensor_to_np01(t_rec)
                ssim_val, psnr_val = compute_cycle_metrics(np_in, np_rec)

                m1, m2, m3 = st.columns(3)
                m1.metric("SSIM", f"{ssim_val:.4f}")
                m2.metric("PSNR (dB)", f"{psnr_val:.2f}")
                m3.metric("Device", str(DEVICE).upper())
            else:
                st.warning("scikit-image is not installed, so SSIM/PSNR metrics are unavailable.")

        st.divider()
        st.subheader("Model Resolution Preview (128x128)")
        strip_cols = st.columns(3)
        labels = [f"Input {input_label}", f"Translated {output_label}", "Reconstructed"]
        tensors = [t_in, t_out, t_rec]
        for col, label, tensor in zip(strip_cols, labels, tensors):
            arr = (tensor_to_np01(tensor) * 255).astype(np.uint8)
            col.image(arr, caption=label, width=128)

    st.divider()
    with st.expander("Q3 Notes"):
        st.markdown(
            """
            - CycleGAN learns mapping without paired images.
            - Cycle consistency enforces A -> B -> A reconstruction.
            - Translation quality depends heavily on checkpoint quality and domain fit.
            """
        )


st.divider()
st.caption(
    "Built with Streamlit | Generative AI Studio | FAST NUCES Spring 2026 | Adnan Ali & Muhammad Suleman"
)