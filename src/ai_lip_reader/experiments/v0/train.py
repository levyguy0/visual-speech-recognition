import os
import random

import torch
from mouth_crop import get_cropped_mouth_video_tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


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

        clip_tensor = get_cropped_mouth_video_tensor(clip_path)

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
for color_dir in os.listdir("data/color_words_only"):
    if color_dir == ".DS_Store":
        continue

    for clip_path in os.listdir(os.path.join("data/color_words_only", color_dir)):
        all_clip_paths.append(
            os.path.join("data/color_words_only", color_dir, clip_path)
        )


random.shuffle(all_clip_paths)
train_clip_paths = all_clip_paths[: int(len(all_clip_paths) * 0.8)]
test_clip_paths = all_clip_paths[int(len(all_clip_paths) * 0.8) :]

print(len(train_clip_paths), len(test_clip_paths))

train_dataset = ColorWordDataset(train_clip_paths)
test_dataset = ColorWordDataset(test_clip_paths)

train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn
)
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
)
clips, lengths, labels = next(iter(train_loader))

print(clips.shape)
print(lengths)
print(labels.shape)
