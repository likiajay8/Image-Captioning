import gradio as gr
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import requests
from io import BytesIO

# -------------------------------
# Load BLIP model (once)
# -------------------------------
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
device = torch.device("cpu")
model.to(device)

# -------------------------------
# Caption generation function
# -------------------------------
def generate_caption(input_image):
    """
    input_image: PIL.Image object or URL string
    """
    # Handle URL input
    if isinstance(input_image, str):
        try:
            response = requests.get(input_image)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            return f"❌ Error loading image from URL: {e}"
    else:
        image = input_image.convert("RGB")
    
    try:
        inputs = processor(images=image, return_tensors="pt")
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
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Textbox(label="Or paste Image URL (optional)")
    ],
    outputs=gr.Textbox(label="Generated Caption"),
    title="🖼️ Captiongen - AI Image Caption Generator",
    description="Upload an image or paste an image URL to generate a descriptive caption using the BLIP model.",
    allow_flagging="never"
)

# -------------------------------
# Launch app
# -------------------------------
iface.launch()
