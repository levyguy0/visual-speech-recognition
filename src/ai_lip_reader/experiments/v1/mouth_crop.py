import cv2
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
    video_tensor = video_tensor.float() / 255.0  # normalize to [0, 1]

    return video_tensor, None
