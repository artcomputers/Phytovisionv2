import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import json

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("PhytovisionModel.pth", map_location=device))
model.to(device)
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dictionary with extra info per class
info_dict = {
    "apple": {
        "Growing Conditions": "Temperate climates; full sun; well-drained soil.",
        "Uses": "Eaten fresh, used in pies, juices, and jams."
    },
    "banana": {
        "Growing Conditions": "Tropical climate; warm and humid; rich soil.",
        "Uses": "Eaten raw, used in smoothies, baking, and chips."
    },
    "beetroot": {
        "Growing Conditions": "Cool weather; loose, fertile soil.",
        "Uses": "Used in salads, juices, and soups."
    },
    "bell pepper": {
        "Growing Conditions": "Warm climate; full sun; moist, well-drained soil.",
        "Uses": "Used in salads, stir-fries, and stuffed dishes."
    },
    "cabbage": {
        "Growing Conditions": "Cool weather; full sun; fertile soil.",
        "Uses": "Used in salads, stews, and sauerkraut."
    },
    "capsicum": {
        "Growing Conditions": "Warm climate; well-drained, fertile soil.",
        "Uses": "Used fresh in salads or cooked in various dishes."
    },
    "carrot": {
        "Growing Conditions": "Cool weather; sandy, loose soil.",
        "Uses": "Used raw, in soups, stews, and juices."
    },
    "cauliflower": {
        "Growing Conditions": "Cool climate; rich, moist soil.",
        "Uses": "Used steamed, roasted, or in curries."
    },
    "chilli pepper": {
        "Growing Conditions": "Warm, sunny climate; well-drained soil.",
        "Uses": "Used to spice up dishes, sauces, and pickles."
    },
    "corn": {
        "Growing Conditions": "Warm climate; full sun; rich soil.",
        "Uses": "Used as food (corn on the cob), flour, and animal feed."
    },
    "cucumber": {
        "Growing Conditions": "Warm climate; lots of sunlight; fertile soil.",
        "Uses": "Used in salads, pickles, and drinks."
    },
    "eggplant": {
        "Growing Conditions": "Warm climate; fertile, well-drained soil.",
        "Uses": "Used in curries, bakes, and stir-fries."
    },
    "garlic": {
        "Growing Conditions": "Cool weather; loose, well-drained soil.",
        "Uses": "Used as seasoning and for medicinal purposes."
    },
    "ginger": {
        "Growing Conditions": "Tropical/subtropical climate; rich, moist soil.",
        "Uses": "Used in teas, cooking, and medicine."
    },
    "grapes": {
        "Growing Conditions": "Warm, dry climate; well-drained soil.",
        "Uses": "Eaten fresh, in wine, or dried as raisins."
    },
    "jalapeno": {
        "Growing Conditions": "Hot, sunny climate; fertile, drained soil.",
        "Uses": "Used in salsas, hot sauces, and stuffing."
    },
    "kiwi": {
        "Growing Conditions": "Temperate climate; well-drained, fertile soil.",
        "Uses": "Eaten fresh or used in desserts and salads."
    },
    "lemon": {
        "Growing Conditions": "Subtropical climate; full sun; acidic soil.",
        "Uses": "Used in drinks, cooking, and cleaning products."
    },
    "lettuce": {
        "Growing Conditions": "Cool weather; moist, rich soil.",
        "Uses": "Used in salads, wraps, and sandwiches."
    },
    "mango": {
        "Growing Conditions": "Tropical/subtropical climate; deep soil.",
        "Uses": "Eaten fresh, in smoothies, and desserts."
    },
    "onion": {
        "Growing Conditions": "Cool-season crop; well-drained soil.",
        "Uses": "Used in cooking, sauces, and salads."
    },
    "orange": {
        "Growing Conditions": "Subtropical to tropical; sandy soil.",
        "Uses": "Eaten fresh, juiced, or used in desserts."
    },
    "paprika": {
        "Growing Conditions": "Warm climate; full sun; fertile soil.",
        "Uses": "Dried and ground for spice."
    },
    "pear": {
        "Growing Conditions": "Temperate climate; loamy, well-drained soil.",
        "Uses": "Eaten fresh, poached, or in desserts."
    },
    "peas": {
        "Growing Conditions": "Cool climate; well-drained soil.",
        "Uses": "Used in soups, curries, and as a side dish."
    },
    "pineapple": {
        "Growing Conditions": "Tropical climate; sandy, acidic soil.",
        "Uses": "Eaten fresh, juiced, or in desserts."
    },
    "pomegranate": {
        "Growing Conditions": "Hot, dry climate; well-drained soil.",
        "Uses": "Eaten fresh, juiced, or used in sauces."
    },
    "potato": {
        "Growing Conditions": "Cool climate; loose, fertile soil.",
        "Uses": "Used boiled, fried, or baked in countless dishes."
    },
    "raddish": {
        "Growing Conditions": "Cool weather; sandy, well-drained soil.",
        "Uses": "Eaten raw or in salads and stir-fries."
    },
    "soy beans": {
        "Growing Conditions": "Warm climate; well-drained soil.",
        "Uses": "Used for soy milk, tofu, and animal feed."
    },
    "tomato": {
        "Growing Conditions": "Warm climate; full sun; fertile soil.",
        "Uses": "Used in sauces, salads, and cooking."
    },
    "spinach": {
        "Growing Conditions": "Cool climate; rich, moist soil.",
        "Uses": "Used in salads, sautés, and smoothies."
    },
    "turnip": {
        "Growing Conditions": "Cool season; loose, fertile soil.",
        "Uses": "Used in soups, stews, and roasted."
    },
    "sweetcorn": {
        "Growing Conditions": "Warm weather; rich, moist soil.",
        "Uses": "Boiled, grilled, or used in salads and chowders."
    },
    "sweetpotato": {
        "Growing Conditions": "Warm climate; sandy, well-drained soil.",
        "Uses": "Baked, mashed, or made into fries and pies."
    },
    "watermelon": {
        "Growing Conditions": "Hot, sunny climate; sandy, well-drained soil.",
        "Uses": "Eaten fresh or in juices and salads."
    }
}

# Streamlit app
st.title("Phytovision v2")
st.write("Upload an image of a fruit or vegetable to classify it.")
st.write("ResNet18 model trained on a fruit and vegetable dataset from Kaggle with the help of PyTorch")
st.write("The dataset used in training of the model can be found at:https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition/")
st.write("Warning: Phytovision is a tool powered by an AI model and may sometimes make mistakes ")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Preprocess image
    image = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        label = class_names[predicted.item()]
    
    st.write(f"### Prediction: {label}")

    if label in info_dict:
        st.subheader("More Information")
        st.markdown(f"**Growing Conditions:** {info_dict[label]['Growing Conditions']}")
        st.markdown(f"**Uses:** {info_dict[label]['Uses']}")
    else:
        st.warning("No additional information available for this item.")
