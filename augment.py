# cnn_full_train_test.py
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image, ImageOps
import os

# -----------------------------
# 1️⃣ Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




print("Device:", device)

# -----------------------------
# 2️⃣ Data augmentasyon
train_transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomAffine(0, translate=(0.1,0.1)),
    transforms.ToTensor()
])

test_transform = transforms.ToTensor()

train_data = datasets.MNIST(root="./data", train=True, download=True, transform=train_transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

test_data = datasets.MNIST(root="./data", train=False, download=True, transform=test_transform)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# -----------------------------
# 3️⃣ CNN Model
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
# 4️⃣ Loss ve optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# 5️⃣ Eğitim
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

# -----------------------------
# 6️⃣ Test doğruluğu
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Test Accuracy: {100*correct/total:.2f}%")

# -----------------------------
# 7️⃣ Modeli kaydet
torch.save(model.state_dict(), "cnn_mnist_aug.pth")
print("Model kaydedildi: cnn_mnist_aug.pth")

# -----------------------------
# 8️⃣ Kendi el yazını test et
test_folder = "my_digits/"  # PNG’lerini buraya koy
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
            img_tensor = img_tensor.to(device)
            output = model(img_tensor)
            _, pred = torch.max(output.data, 1)
        print(f"{filename} -> Tahmin: {pred.item()}")