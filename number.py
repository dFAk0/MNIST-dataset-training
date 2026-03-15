# test_5digit_with_cnnmnistaug.py
import torch
from torchvision import transforms
from PIL import Image, ImageOps
import os
import torch.nn as nn

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -----------------------------
# Model tanımı (CNNMnistAug ile uyumlu)
class CNNMnistAug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7,128),
            nn.ReLU(),
            nn.Linear(128,10)
        )

    def forward(self,x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# -----------------------------
# Modeli yükle
model = CNNMnistAug().to(device)
model.load_state_dict(torch.load("cnn_mnist_aug.pth", map_location=device))
model.eval()

# -----------------------------
# Transform
transform = transforms.Compose([
    transforms.ToTensor(),
])

# -----------------------------
# 5 basamaklı sayı resimlerini test et
test_folder = "my_digits_test"

def predict_digit(img):
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        pred = output.argmax(1).item()
    return str(pred)

for fname in os.listdir(test_folder):
    if fname.endswith(".png"):
        img_path = os.path.join(test_folder,fname)
        img = Image.open(img_path).convert('L')
        img = ImageOps.invert(img)
        w, h = img.size

        # 5 eşit parçaya böl
        digit_width = w // 4
        pred_number = ""
        for i in range(4):
            left = i*digit_width
            right = (i+1)*digit_width
            digit_img = img.crop((left, 0, right, h)).resize((28,28))
            pred_digit = predict_digit(digit_img)
            pred_number += pred_digit

        print(f"{fname} -> Tahmin: {pred_number}")