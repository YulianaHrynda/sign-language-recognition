import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from tqdm import tqdm

def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss, correct = 0.0, 0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    accuracy = 100. * correct / len(train_loader.dataset)
    print(f"Train Loss: {running_loss / len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")
    return accuracy


def main():
    data_dir = "archive/asl_alphabet_train"
    num_epochs = 10
    batch_size = 32
    learning_rate = 0.0005
    num_classes = 29

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageFolder(root=data_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, criterion, optimizer, epoch)

    torch.save(model.state_dict(), "asl_resnet18.pth")
    print("Model saved as asl_resnet18.pth")


if __name__ == "__main__":
    main()
