import os

import cv2

TARGET_DIR = "data/color_words_only"
SOURCE_DIR = "data/videos"
ALIGN_DIR = "data/alignments"

speakers = [f"s{i}" for i in range(1, 26) if i != 21]

colors = ["blue", "green", "red", "white"]
vocab = {color: i for i, color in enumerate(colors)}

for speaker in speakers:
    SPEAKER_SOURCE_DIR = f"{SOURCE_DIR}/{speaker}"
    SPEAKER_ALIGN_DIR = f"{ALIGN_DIR}/{speaker}"

    for file_name in os.listdir(SPEAKER_SOURCE_DIR):
        if not file_name.endswith(".mpg"):
            continue

        print(f"Processing {file_name}")
        video_path = f"{SPEAKER_SOURCE_DIR}/{file_name}"
        align_path = f"{SPEAKER_ALIGN_DIR}/{file_name.split('.')[0]}.align"

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        with open(align_path) as f:
            lines = f.readlines()

            for line in lines:
                start, end, word = line.strip().split()

                if word not in vocab:
                    continue

                start = int(start)
                end = int(end)

                start_frame = int(start / 75000 * total_frames)
                end_frame = int(end / 75000 * total_frames)

                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

                word_frames = []

                for _ in range(end_frame - start_frame):
                    success, frame = cap.read()

                    if not success:
                        break

                    word_frames.append(frame)

                if not word_frames:
                    print(f"No frames found for word {word}")
                    continue

                height, width = word_frames[0].shape[:2]

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]

                os.makedirs(f"{TARGET_DIR}/{speaker}/{word}", exist_ok=True)

                writer = cv2.VideoWriter(
                    f"{TARGET_DIR}/{speaker}/{word}/{file_name.split('.')[0]}.mp4",
                    fourcc,
                    25,
                    (width, height),
                )

                for frame in word_frames:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    writer.write(gray)

                writer.release()

        cap.release()
