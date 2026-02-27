🖼️ CaptionGen — AI Image Caption Generator

An AI-powered web app that generates descriptive captions for images using the BLIP (Bootstrapping Language-Image Pretraining) model from Salesforce.
Built with Transformers, PyTorch, and Gradio for an interactive interface.


🚀 Features

🧠 State-of-the-art image captioning using BLIP

🖼 Upload any image and get an automatic description

⚡ Runs on CPU or GPU (auto-detect)

🌐 Simple and interactive Gradio UI

🪶 Lightweight and easy to deploy


🛠 Tech Stack

Python

PyTorch

Hugging Face Transformers

BLIP Image Captioning Model

Gradio

Pillow


⚙️ How It Works

User uploads an image

Image is processed using BlipProcessor

BLIP model generates caption tokens

Tokens are decoded into natural language text

Caption is displayed in the UI


📂 Project Structure
captiongen/

├── app.py               # Main Gradio app

├── requirements.txt     # Dependencies

└── README.md            # Documentation


▶️ Run Locally
# Clone repo
git clone https://github.com/your-username/captiongen.git

# Enter folder
cd captiongen

# Install dependencies
pip install -r requirements.txt

# Run app
python app.py


📦 Requirements

Your requirements.txt is correct ✅

gradio==3.43.0

torch==2.1.0

torchvision==0.16.0

torchaudio==2.1.0

transformers==4.35.0

pillow<11.0,>=8.0


🎯 Use Cases

Accessibility tools for visually impaired users

Image search and tagging

Social media automation

AI learning and experimentation


🔮 Future Improvements

Add multiple caption options

Support batch image uploads

Deploy on Hugging Face Spaces

Add caption confidence scores

Integrate with vision-language chat


👨‍💻 Author

Likith H P
