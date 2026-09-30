import os
from pathlib import Path

import cv2
import face_alignment
import numpy as np
import torch


def get_cropped_mouth_video_tensor(fa, video_path, height=96, width=128):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        success, frame = cap.read()
        if not success:
            break
        frames.append(frame)

    cap.release()

    if not frames:
        raise RuntimeError("Could not read video")

    mouth_cropped_frames = []
    for frame in frames:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = fa.get_landmarks(frame_rgb)

        if faces is None:
            print("No face detected")
        else:
            landmarks = faces[0]
            mouth = landmarks[48:68]

            xs = mouth[:, 0]
            ys = mouth[:, 1]
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            padding = 10
            mouth_crop = frame_rgb[
                int(y_min) - padding : int(y_max) + padding,
                int(x_min) - padding : int(x_max) + padding,
            ]
            # resize to 128x96
            mouth_crop = cv2.resize(mouth_crop, (128, 96))
            mouth_cropped_frames.append(mouth_crop)

    # convert to video tensor
    if len(mouth_cropped_frames) == 0:
        print("No mouth cropped frames")
        return None, "No mouth cropped frames"

    video = np.stack(mouth_cropped_frames)
    video_tensor = torch.from_numpy(video)
    video_tensor = video_tensor.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
    # Keep uint8 pixels for compact storage; normalize when loading for training.

    return video_tensor, None


input_root = Path("data/all_grid_word_clips")
output_root = Path("data/all_grid_word_processed")

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device="cuda")

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
