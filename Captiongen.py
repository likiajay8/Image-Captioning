import gradio as gr
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# -------------------------------
# Load BLIP model (once)
# -------------------------------
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Use CPU or GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -------------------------------
# Caption generation function
# -------------------------------
def generate_caption(uploaded_image):
    """
    input_image: PIL.Image object
    """
    if uploaded_image is None:
        return "❌ No image provided."

    try:
        image = uploaded_image.convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}  # move inputs to same device
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        return f"❌ Error generating caption: {e}"

# -------------------------------
# Gradio Interface
# -------------------------------
iface = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Textbox(label="Generated Caption"),
    title="🖼️ Captiongen - AI Image Caption Generator",
    description="Upload an image to generate a descriptive caption using the BLIP model.",
    allow_flagging="never"
)

# -------------------------------
# Launch app
# -------------------------------
iface.launch()
