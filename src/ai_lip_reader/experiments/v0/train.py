import os
import random

import torch
from mouth_crop import get_cropped_mouth_video_tensor
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class ColorWordDataset(Dataset):
    def __init__(self, clip_paths):
        self.clip_paths = clip_paths

        color_to_label = {
            "blue": 0,
            "green": 1,
            "red": 2,
            "white": 3,
        }
        self.clip_path_to_label = {
            clip_path: color_to_label[clip_path.split("/")[-2]]
            for clip_path in clip_paths
        }

    def __len__(self):
        return len(self.clip_paths)

    def __getitem__(self, idx):
        clip_path = self.clip_paths[idx]
        label = self.clip_path_to_label[clip_path]

        clip_tensor = torch.load(clip_path)

        label_tensor = torch.tensor(label, dtype=torch.long)
        return clip_tensor, label_tensor


def collate_fn(batch):
    clips, labels = zip(*batch)

    lengths = torch.tensor([len(clip) for clip in clips])

    clips = pad_sequence(
        clips,
        batch_first=True,
        padding_value=0.0,
    )

    labels = torch.stack(labels)

    return clips, lengths, labels


all_clip_paths = []

for color in ["blue", "green", "red", "white"]:
    directory = f"data/color_words_processed/{color}"

    for filename in os.listdir(directory):
        if filename.endswith(".pt"):
            all_clip_paths.append(os.path.join(directory, filename))

random.shuffle(all_clip_paths)
train_clip_paths = all_clip_paths[: int(len(all_clip_paths) * 0.8)]
test_clip_paths = all_clip_paths[int(len(all_clip_paths) * 0.8) :]

train_dataset = ColorWordDataset(train_clip_paths)
test_dataset = ColorWordDataset(test_clip_paths)

train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn
)
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
)


class LipReader(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            batch_first=True,
        )

        self.classifier = nn.Linear(128, 4)

    def forward(self, x, lengths):
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        x = self.cnn(x)
        x = x.flatten(1)
        x = x.reshape(B, T, 128)
        x = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(x)
        logits = self.classifier(hidden[-1])
        return logits


model = LipReader().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    train_avg_loss = 0
    for clips, lengths, labels in train_loader:
        clips, labels = clips.to(device), labels.to(device)
        logits = model(clips, lengths)
        loss = loss_fn(logits, labels)
        train_avg_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_avg_loss /= len(train_loader)

    model.eval()
    test_avg_loss = 0
    with torch.inference_mode():
        for clips, lengths, labels in test_loader:
            clips, labels = (
                clips.to(device),
                labels.to(device),
            )
            logits = model(clips, lengths)
            loss = loss_fn(logits, labels)
            test_avg_loss += loss.item()
        test_avg_loss /= len(test_loader)

    print(
        f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_avg_loss:.4f}, Test Loss: {test_avg_loss:.4f}"
    )
