import cv2
import face_alignment
import numpy as np

cap = cv2.VideoCapture("data/s1/bbaf2n.mpg")

success, frame = cap.read()
cap.release()

if not success:
    raise RuntimeError("Could not read video")

frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device="mps")

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
    padding = 5
    mouth_crop = frame[
        int(y_min) - padding : int(y_max) + padding,
        int(x_min) - padding : int(x_max) + padding,
    ]
    # resize to 128x96
    mouth_crop = cv2.resize(mouth_crop, (128, 96))
    cv2.imshow("Mouth crop", mouth_crop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
