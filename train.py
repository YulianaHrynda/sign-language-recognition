import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from tqdm import tqdm
import kagglehub


def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss, correct = 0.0, 0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    accuracy = 100. * correct / len(train_loader.dataset)
    print(f"Train Loss: {running_loss / len(train_loader):.4f}, Train Accuracy: {accuracy:.2f}%")
    return accuracy


def validate(model, device, val_loader, criterion):
    model.eval()
    val_loss, correct = 0.0, 0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

    accuracy = 100. * correct / len(val_loader.dataset)
    print(f"Val Loss: {val_loss / len(val_loader):.4f}, Val Accuracy: {accuracy:.2f}%")
    return accuracy


def main():
    # 🧠 Download ASL dataset using kagglehub
    path = kagglehub.dataset_download("grassknoted/asl-alphabet")
    data_dir = os.path.join(path, "asl_alphabet_train")  # contains class folders

    num_epochs = 3
    batch_size = 32
    learning_rate = 0.0005
    num_classes = 29
    val_split = 0.2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    full_dataset = ImageFolder(root=data_dir, transform=transform)
    val_size = int(val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, criterion, optimizer, epoch)
        validate(model, device, val_loader, criterion)

    torch.save(model.state_dict(), "asl_resnet18.pth")
    print("✅ Model saved as asl_resnet18.pth")


if __name__ == "__main__":
    main()

