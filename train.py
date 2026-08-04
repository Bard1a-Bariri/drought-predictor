import os
import random
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import DroughtModelResNet

def prepare_dataset(source_dir, target_dir, val_ratio=0.2):
    if os.path.exists(os.path.join(target_dir, "train")) and os.path.exists(
        os.path.join(target_dir, "val")
    ):
        print(f"Dataset already organized in '{target_dir}'. Skipping dataset creation.")
        return

    if not os.path.exists(source_dir):
        raise FileNotFoundError(
            f"Could not find organized '{target_dir}' or raw folder '{source_dir}'. "
            f"Please place your unzipped dataset into '{source_dir}'."
        )

    print(f"Organizing raw data from '{source_dir}' into '{target_dir}'...")
    random.seed(42)

    for split in ["train", "val"]:
        for label in ["0_healthy", "1_drought"]:
            os.makedirs(os.path.join(target_dir, split, label), exist_ok=True)

    for folder_name in os.listdir(source_dir):
        folder_path = os.path.join(source_dir, folder_name)

        if os.path.isdir(folder_path):
            target_label = (
                "0_healthy" if "healthy" in folder_name.lower() else "1_drought"
            )

            images = [
                f
                for f in os.listdir(folder_path)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))
            ]
            random.shuffle(images)

            val_count = int(len(images) * val_ratio)
            val_images = images[:val_count]
            train_images = images[val_count:]

            for img in val_images:
                src = os.path.join(folder_path, img)
                dst = os.path.join(
                    target_dir, "val", target_label, f"{folder_name}_{img}"
                )
                shutil.copy(src, dst)

            for img in train_images:
                src = os.path.join(folder_path, img)
                dst = os.path.join(
                    target_dir, "train", target_label, f"{folder_name}_{img}"
                )
                shutil.copy(src, dst)

def main():
    RAW_DATASET_DIR = "ground_water_stress_raw"  
    DATASET_DIR = "dataset"
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 10
    SAVE_FILENAME = "ground_crop_model.pth"


    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_path = os.path.join(DATASET_DIR, "train")
    val_path = os.path.join(DATASET_DIR, "val")

    train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_path, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DroughtModelResNet().to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.model.fc.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device) 
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_dataset)

        model.eval()
        val_corrects = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                preds = (outputs > 0.5).squeeze().long()
                val_corrects += torch.sum(preds == labels.data)

        val_acc = val_corrects.double() / len(val_dataset)

        print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {epoch_loss:.4f} | Val Acc: {val_acc * 100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_FILENAME)

if __name__ == "__main__":
    main()