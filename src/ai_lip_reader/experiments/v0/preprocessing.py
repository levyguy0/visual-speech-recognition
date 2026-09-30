import os
from pathlib import Path

import face_alignment
import torch
from mouth_crop import get_cropped_mouth_video_tensor

input_root = Path("data/color_words_only")
output_root = Path("data/color_words_processed")

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device="mps")

# for speaker in ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10"]:
for speaker in [
    "s11",
    "s12",
    "s13",
    "s14",
    "s15",
    "s16",
    "s17",
    "s18",
    "s19",
    "s20",
    "s22",
    "s23",
    "s24",
    "s25",
]:
    for color in ["blue", "green", "red", "white"]:
        input_dir = input_root / speaker / color
        output_dir = output_root / speaker / color

        output_dir.mkdir(parents=True, exist_ok=True)

        for filename in os.listdir(input_dir):
            if not filename.endswith(".mp4"):
                continue

            input_path = input_dir / filename
            output_path = output_dir / f"{input_path.stem}.pt"

            print(f"Processing {input_path}")

            clip_tensor, error = get_cropped_mouth_video_tensor(fa, str(input_path))
            if error:
                print(error)
                continue
            else:
                torch.save(clip_tensor, output_path)
