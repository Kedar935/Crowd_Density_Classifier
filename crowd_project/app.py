# import streamlit as st
# import torch
# from torchvision import transforms, models, datasets
# from PIL import Image
# import torch.nn as nn

# # Config
# MODEL_PATH = "crowd_classifier.pth"
# DATASET_PATH = "dataset/train"

# # Load dataset for class names
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])
# ])
# train_data = datasets.ImageFolder(DATASET_PATH, transform=transform)
# class_names = train_data.classes

# # Load model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = models.resnet18(pretrained=False)
# model.fc = nn.Linear(model.fc.in_features, len(class_names))
# model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
# model.to(device)
# model.eval()

# # Prediction function
# def predict_image(pil_img):
#     img = transform(pil_img).unsqueeze(0).to(device)
#     with torch.no_grad():
#         outputs = model(img)
#         probs = torch.softmax(outputs, dim=1)[0]
#         _, preds = torch.max(outputs, 1)
#     return class_names[preds.item()], probs.cpu().numpy()

# # Streamlit UI
# st.set_page_config(page_title="Crowd Density Classifier", layout="wide")

# st.markdown("<h1 style='text-align:center;'>🧑‍🤝‍🧑 Crowd Density Classifier</h1>", unsafe_allow_html=True)
# st.write("<p style='text-align:center;'>Upload an image to classify it as <b>Sparse</b> or <b>Dense</b> crowd.</p>", unsafe_allow_html=True)

# uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# if uploaded_file:
#     img = Image.open(uploaded_file).convert("RGB")
#     label, probs = predict_image(img)

#     col1, col2 = st.columns([2, 1])

#     with col1:
#         st.image(img, caption="Uploaded Image", use_container_width=True)

#     with col2:
#         # Prediction Badge
#         color = "#4CAF50" if label.lower() == "sparse" else "#FF5733"
#         st.markdown(f"""
#         <div style="background-color:{color};padding:10px;border-radius:10px;text-align:center;">
#             <h2 style="color:white;">Prediction: {label.upper()}</h2>
#         </div>
#         """, unsafe_allow_html=True)

#         # Probabilities
#         st.subheader("📊 Class Probabilities")
#         for name, prob in zip(class_names, probs):
#             st.markdown(f"<b>{name.capitalize()}</b>: {prob*100:.2f}%", unsafe_allow_html=True)
#             st.progress(float(prob))

import streamlit as st
import torch
from torchvision import transforms, models, datasets
from PIL import Image
import torch.nn as nn
import time
import os

# --- Config ---
MODEL_PATH = "crowd_classifier.pth"
DATASET_PATH = "dataset/train"

# --- Page Setup ---
st.set_page_config(page_title="Crowd Density Classifier", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.title("📁 Upload Image")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    st.markdown("---")
    st.markdown("🔍 This app classifies crowd density as **Sparse** or **Dense** using AI.")
    #st.markdown("📌 Powered by PyTorch & Streamlit")

# --- Load Dataset for Class Names ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
train_data = datasets.ImageFolder(DATASET_PATH, transform=transform)
class_names = train_data.classes

# --- Load Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# --- Prediction Function ---
def predict_image(pil_img):
    img = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)[0]
        _, preds = torch.max(outputs, 1)
    return class_names[preds.item()], probs.cpu().numpy()

# --- Main Content ---
st.markdown("<h1 style='text-align:center;'>🧑‍🤝‍🧑 Crowd Density Classifier</h1>", unsafe_allow_html=True)
st.write("<p style='text-align:center;'>Upload an image from the sidebar to classify it.</p>", unsafe_allow_html=True)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    label, probs = predict_image(img)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.image(img, caption="Uploaded Image", use_container_width=True)

    with col2:
        # Prediction Badge
        color = "#4CAF50" if label.lower() == "sparse" else "#FF5733"
        st.markdown(f"""
        <div style="background-color:{color};padding:15px;border-radius:10px;text-align:center;">
            <h2 style="color:white;">Prediction: {label.upper()}</h2>
        </div>
        """, unsafe_allow_html=True)

        # Class Probabilities with Animation
        st.subheader("📊 Class Probabilities")
        for name, prob in zip(class_names, probs):
            st.markdown(f"**{name.capitalize()}**")
            progress_bar = st.progress(0)
            for i in range(int(prob * 100)):
                time.sleep(0.005)
                progress_bar.progress(i + 1)


