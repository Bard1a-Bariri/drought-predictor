import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
import tensorflow as tf
from model import SatelliteDroughtModel
import numpy as np

def parse_tfrecord(example_proto):
    feature_description = {
        f'B{i}': tf.io.FixedLenFeature([], tf.string) for i in range(1, 11)
    }
    feature_description['label'] = tf.io.FixedLenFeature([], tf.int64)

    parsed = tf.io.parse_single_example(example_proto, feature_description)

    bands = []
    for i in range(1, 11):
        band_raw = tf.io.decode_raw(parsed[f'B{i}'], tf.uint8)
        band = tf.reshape(band_raw, [65, 65])
        band = tf.cast(band, tf.float32) / 255.0
        bands.append(band)

    image = tf.stack(bands, axis=0)
    label = parsed['label']

    return image.numpy(), label.numpy()

class DroughtWatchTFRecordDataset(IterableDataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __iter__(self):
        raw_dataset = tf.data.TFRecordDataset(self.file_paths)
        for raw_record in raw_dataset:
            image, label = parse_tfrecord(raw_record)
            yield torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def train_satellite():
    BATCH_SIZE = 64
    EPOCHS = 10
    LEARNING_RATE = 0.0005
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    train_dir = "dw_data/droughtwatch_data/train"
    val_dir = "dw_data/droughtwatch_data/val"

    train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, f))]
    val_files = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if os.path.isfile(os.path.join(val_dir, f))]

    print(f"Loaded {len(train_files)} train partition files and {len(val_files)} val partition files.")

    train_dataset = DroughtWatchTFRecordDataset(train_files)
    val_dataset = DroughtWatchTFRecordDataset(val_files)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = SatelliteDroughtModel(in_channels=10, num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        total_samples = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

        epoch_loss = running_loss / max(total_samples, 1)
        print(f"Epoch {epoch:02d}/{EPOCHS:02d} | Train Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), "satellite_droughtwatch.pth")
    print("Saved satellite_droughtwatch.pth successfully!")

if __name__ == "__main__":
    train_satellite()