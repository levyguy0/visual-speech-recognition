import os
from pathlib import Path

import face_alignment
import torch
from mouth_crop import get_cropped_mouth_video_tensor

input_root = Path("data/all_grid_word_clips")
output_root = Path("data/all_grid_word_processed")

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device="mps")

speakers = [f"s{i}" for i in range(1, 26) if i != 21]
for speaker in speakers:
    for word in [dir for dir in os.listdir(input_root / speaker) if dir != ".DS_Store"]:
        input_dir = input_root / speaker / word
        output_dir = output_root / speaker / word

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
