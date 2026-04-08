import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from streamlit_drawable_canvas import st_canvas

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Edge2Shoes – Pix2Pix",
    page_icon="👟",
    layout="centered",
)

# ── Model Architecture (copy from notebook) ──────────────────────────────────
class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)]
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
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
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
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False)
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
            nn.ConvTranspose2d(128, out_channels, 4, stride=2, padding=1),
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


# ── Load model (cached) ───────────────────────────────────────────────────────
@st.cache_resource
def load_generator(checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = UNetGenerator().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    # Handle full checkpoint dict or bare state_dict
    if isinstance(state, dict) and "generator" in state:
        gen.load_state_dict(state["generator"])
    else:
        gen.load_state_dict(state)
    gen.eval()
    return gen, device


# ── Inference helper ──────────────────────────────────────────────────────────
def predict(generator, device, pil_image: Image.Image) -> Image.Image:
    """Run generator on a PIL edge image; return PIL shoe image."""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    tensor = transform(pil_image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        out = generator(tensor)
    out = out.squeeze(0).cpu().permute(1, 2, 0).numpy()
    out = ((out + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out)


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("👟 Edge2Shoes — Pix2Pix Demo")
st.markdown("Upload your **trained generator weights** (`.pth`), draw or upload a **shoe edge map**, then click **Generate**.")

# --- Sidebar: model upload ---------------------------------------------------
st.sidebar.header("⚙️ Model")
model_file = st.sidebar.file_uploader(
    "Upload generator_final.pth",
    type=["pth"],
    help="Upload the generator weights saved from your Colab training.",
)

generator = None
device = None

if model_file is not None:
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp:
        tmp.write(model_file.read())
        tmp_path = tmp.name
    try:
        generator, device = load_generator(tmp_path)
        st.sidebar.success("✅ Model loaded!")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to load model: {e}")
else:
    st.sidebar.info("Please upload your `.pth` weights to enable generation.")

# --- Input tab: draw or upload -----------------------------------------------
tab_draw, tab_upload = st.tabs(["✏️ Draw Edge", "📂 Upload Image"])

input_image: Image.Image | None = None

with tab_draw:
    st.markdown("Draw a **black shoe outline on white background** (256 × 256).")
    canvas_result = st_canvas(
        fill_color="rgba(255,255,255,0)",
        stroke_width=4,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=256,
        width=256,
        drawing_mode="freedraw",
        key="canvas",
    )
    if canvas_result.image_data is not None:
        arr = canvas_result.image_data.astype(np.uint8)
        input_image = Image.fromarray(arr).convert("RGB")

with tab_upload:
    uploaded = st.file_uploader("Upload a 256×256 edge map (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded:
        input_image = Image.open(uploaded).convert("RGB")
        st.image(input_image, caption="Uploaded edge map", width=256)

# --- Generate ----------------------------------------------------------------
st.divider()
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Edge")
    if input_image is not None:
        st.image(input_image, width=256)
    else:
        st.info("Draw or upload an edge map above.")

with col2:
    st.subheader("Generated Shoe")
    if st.button("🎨 Generate Shoe", type="primary", disabled=(generator is None or input_image is None)):
        with st.spinner("Generating…"):
            result = predict(generator, device, input_image)
        st.image(result, width=256)
        # Download button
        import io
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        st.download_button("⬇️ Download", buf.getvalue(), "generated_shoe.png", "image/png")
    elif generator is None:
        st.warning("Upload model weights first.")
    elif input_image is None:
        st.warning("Provide an edge map first.")

# --- Footer ------------------------------------------------------------------
st.divider()
st.caption("Pix2Pix U-Net + PatchGAN · trained on edges2shoes dataset")
