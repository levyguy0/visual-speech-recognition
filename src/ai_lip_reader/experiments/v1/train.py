import os
import random

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class GRIDWordDataset(Dataset):
    def __init__(self, clip_paths, vocab):
        self.clip_paths = clip_paths
        self.vocab = vocab
        word_to_label = {v: i for i, v in enumerate(vocab)}
        self.clip_path_to_label = {
            clip_path: word_to_label[clip_path.split("/")[-2]]
            for clip_path in clip_paths
        }

    def __len__(self):
        return len(self.clip_paths)

    def __getitem__(self, idx):
        clip_path = self.clip_paths[idx]
        label = self.clip_path_to_label[clip_path]

        clip_tensor = torch.load(clip_path, weights_only=True)
        if clip_tensor.dtype == torch.uint8:
            clip_tensor = clip_tensor.float() / 255.0

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


train_clip_paths = []
test_clip_paths = []
vocab = []

for speaker in [s for s in os.listdir("data/videos") if s != ".DS_Store"]:
    for word in [
        d
        for d in os.listdir(f"data/all_grid_word_processed/{speaker}")
        if d != ".DS_Store"
    ]:
        directory = f"data/all_grid_word_processed/{speaker}/{word}"

        if speaker in ["s21", "s23", "s24", "s25"]:
            for filename in os.listdir(directory):
                if filename.endswith(".pt"):
                    test_clip_paths.append(os.path.join(directory, filename))
        else:
            for filename in os.listdir(directory):
                if filename.endswith(".pt"):
                    train_clip_paths.append(os.path.join(directory, filename))

        if word not in vocab:
            vocab.append(word)

random.shuffle(train_clip_paths)
random.shuffle(test_clip_paths)

print(f"Train clip paths: {len(train_clip_paths)}")
print(f"Test clip paths: {len(test_clip_paths)}")

train_dataset = GRIDWordDataset(train_clip_paths, vocab)
test_dataset = GRIDWordDataset(test_clip_paths, vocab)

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

        self.dropout = nn.Dropout(p=0.3)
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
        logits = self.classifier(self.dropout(hidden[-1]))
        return logits


model = LipReader().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

num_epochs = 25
best_model_state = None
lowest_test_loss = float("inf")

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
    test_correct = 0
    test_total = 0
    test_avg_loss = 0
    with torch.inference_mode():
        for clips, lengths, labels in test_loader:
            clips, labels = (
                clips.to(device),
                labels.to(device),
            )
            logits = model(clips, lengths)
            y_pred_probs = torch.softmax(logits, dim=1)
            y_pred = torch.argmax(y_pred_probs, dim=1)

            test_correct += (y_pred == labels).sum().item()
            test_total += labels.size(0)
            loss = loss_fn(logits, labels)

            if loss.item() < lowest_test_loss:
                lowest_test_loss = loss.item()
                best_model_state = model.state_dict()

            test_avg_loss += loss.item()
        test_avg_loss /= len(test_loader)
        test_accuracy = test_correct / test_total

    print(
        f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_avg_loss:.4f}, Test Loss: {test_avg_loss:.4f}, Test Accuracy: {test_accuracy:.2f}"
    )

# save best model
torch.save(best_model_state, "best_model.pth")

model.eval()

all_labels = []
all_predictions = []

with torch.inference_mode():
    for clips, lengths, labels in test_loader:
        clips = clips.to(device)
        labels = labels.to(device)

        logits = model(clips, lengths)
        predictions = logits.argmax(dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())

class_names = ["blue", "green", "red", "white"]

cm = confusion_matrix(all_labels, all_predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names,
)

disp.plot()
plt.title("Lip Reader Confusion Matrix")
plt.show()
