
import torch
from torch import nn
from torchvision import transforms
from PIL import Image, ImageOps
import os

# -----------------------------
# 1️⃣ Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -----------------------------
# 2️⃣ CNN Model (aynı yapıyı kullan)
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

model = CNN().to(device)

# -----------------------------
# 3️⃣ Model ağırlıklarını yükle
model.load_state_dict(torch.load("cnn_mnist_aug.pth", map_location=device))
model.eval()

# -----------------------------
# 4️⃣ Kendi el yazını test et
test_folder = "numdata/"  # PNG’lerini buraya koy
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28,28)),
    transforms.ToTensor()
])

for filename in os.listdir(test_folder):
    if filename.endswith(".png"):
        img_path = os.path.join(test_folder, filename)
        img = Image.open(img_path).convert('L')
        img = ImageOps.invert(img)  # MNIST formatına uygun
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            _, pred = torch.max(output.data, 1)
        print(f"{filename} -> Tahmin: {pred.item()}")