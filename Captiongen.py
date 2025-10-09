import os

# -------------------------------
# Install PyTorch on Streamlit Cloud
# -------------------------------
os.system("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")

# -------------------------------
# Imports
# -------------------------------
import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import requests
from io import BytesIO

# -------------------------------
# Streamlit page setup
# -------------------------------
st.set_page_config(page_title="AI Image Caption Generator", page_icon="🖼️")
st.title("🖼️ AI Image Caption Generator")
st.write("Generate captions from an image via URL or by uploading a local file.")

# -------------------------------
# Load model (cached)
# -------------------------------
@st.cache_resource(show_spinner=True)
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    device = torch.device("cpu")
    model.to(device)
    return processor, model, device

processor, model, device = load_model()

# -------------------------------
# Image input section
# -------------------------------
input_choice = st.radio("Choose input type:", ("URL", "Local File"))
image = None

if input_choice == "URL":
    image_url = st.text_input("Paste the image URL here:")
    if image_url:
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            st.image(image, caption="Uploaded Image from URL", use_column_width=True)
        except Exception as e:
            st.error(f"❌ Error loading image from URL: {e}")

elif input_choice == "Local File":
    uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png", "bmp", "gif"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Local Image", use_column_width=True)

# -------------------------------
# Generate caption
# -------------------------------
if image is not None:
    if st.button("Generate Caption"):
        with st.spinner("Generating caption... Please wait ⏳"):
            try:
                inputs = processor(images=image, return_tensors="pt")
                out = model.generate(**inputs)
                caption = processor.decode(out[0], skip_special_tokens=True)
                st.success(f"✅ Caption: {caption}")
            except Exception as e:
                st.error(f"❌ Error generating caption: {e}")
