# ImplementingGenerative Models:
# a) Autoencoder for image reconstruction
# b) Word Prediction using RNN
# c) Image Captioning *

# !pip install torch
# !pip install torchvision
# !pip install transformers
# !pip install nltk


import torch
from torchvision.transforms import transforms
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import nltk
nltk.download('punkt')

# Load the pre-trained image captioning model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load and preprocess the image
image_path = "image.png"
image = Image.open(image_path).convert("RGB")  # Convert to RGB
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
input_tensor = preprocess(image).unsqueeze(0)

# Generate captions
with torch.no_grad():
    captions = model.generate(pixel_values=input_tensor)

# Decode the generated captions
caption_text = processor.decode(captions[0], skip_special_tokens=True)

# Print the generated caption
print("Generated Caption:", caption_text)