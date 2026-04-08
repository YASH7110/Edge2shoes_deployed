import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from streamlit_drawable_canvas import st_canvas

import gdown
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Edge2Shoes – Pix2Pix",
    page_icon="👟",
    layout="centered",
)

# ── Download model if not exists ─────────────────────────────────────────────
MODEL_PATH = "generator.pth"

if not os.path.exists(MODEL_PATH):
    with st.spinner("📥 Downloading model (first time only)..."):
        url = "https://drive.google.com/uc?id=1QzYSbFzenUmuGXH2f6ltCg8xb5oo_ITN"
        gdown.download(url, MODEL_PATH, quiet=False)

# ── Model Architecture ───────────────────────────────────────────────────────
class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        return torch.cat((x, skip_input), 1)


class UNetGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = UNetDown(3, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final_up(u7)


# ── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_generator():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetGenerator().to(device)
    state = torch.load(MODEL_PATH, map_location=device)

    if isinstance(state, dict) and "generator" in state:
        model.load_state_dict(state["generator"])
    else:
        model.load_state_dict(state)

    model.eval()
    return model, device


# Load model once
generator, device = load_generator()

st.sidebar.success("✅ Model loaded automatically!")

# ── Inference ────────────────────────────────────────────────────────────────
def predict(generator, device, image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = generator(tensor)

    output = output.squeeze().cpu().permute(1, 2, 0).numpy()
    output = ((output + 1) / 2 * 255).clip(0, 255).astype(np.uint8)

    return Image.fromarray(output)


# ── UI ───────────────────────────────────────────────────────────────────────
st.title("👟 Edge2Shoes — Pix2Pix Demo")
st.markdown("Draw or upload a shoe edge → generate realistic shoe image")

tab1, tab2 = st.tabs(["✏️ Draw", "📂 Upload"])

input_image = None

with tab1:
    canvas = st_canvas(
        stroke_width=4,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=256,
        width=256,
        drawing_mode="freedraw",
    )
    if canvas.image_data is not None:
        input_image = Image.fromarray(canvas.image_data.astype(np.uint8)).convert("RGB")

with tab2:
    file = st.file_uploader("Upload edge image", type=["png", "jpg"])
    if file:
        input_image = Image.open(file).convert("RGB")
        st.image(input_image, width=256)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input")
    if input_image:
        st.image(input_image, width=256)

with col2:
    st.subheader("Output")
    if st.button("Generate"):
        if input_image:
            result = predict(generator, device, input_image)
            st.image(result, width=256)

            import io
            buf = io.BytesIO()
            result.save(buf, format="PNG")
            st.download_button("Download", buf.getvalue(), "shoe.png")
        else:
            st.warning("Provide input first")
