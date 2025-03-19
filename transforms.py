import torch
import torchvision
from torchvision import models
from torchvision import transforms
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Устройство: {device}")

# Загрузка данных
train_dataset = ImageFolder("dataset/train", transform=train_transforms)
val_dataset = ImageFolder("dataset/val", transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Загрузка модели ResNet-18
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features  # Размер входа последнего слоя
model.fc = nn.Linear(num_ftrs, 2)  # Новый слой с 2 классами
model = model.to(device)

# Оптимизатор и функция потерь
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Функция обучения
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        print(f"Эпоха {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Средняя потеря на обучении: {running_loss / len(train_loader)}")

        # Оценка на валидации
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Точность на валидации: {accuracy:.2f}%")

# Запуск обучения
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)

torch.save(model.state_dict(), "resnet_defect_classifier.pth")

model.load_state_dict(torch.load("resnet_defect_classifier.pth"))
model.eval()

