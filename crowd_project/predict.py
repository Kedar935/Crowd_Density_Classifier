import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

# Paths
model_path = "crowd_classifier.pth"
dataset_path = "dataset/train"  # Needed for class names

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load dataset for class names
from torchvision import datasets
train_data = datasets.ImageFolder(dataset_path, transform=transform)

# Load model
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Predict function
def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)
    return train_data.classes[preds.item()]

# Example
test_img = "test2.jpeg"
result = predict_image(test_img)
print(f"Prediction: {result}")
