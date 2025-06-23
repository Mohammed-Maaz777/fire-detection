import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # 🔁 Add this

import cv2
import torch
import numpy as np
import importlib.util
from yolov5.utils.general import non_max_suppression



# Set root directory to yolov5
ROOT = Path(__file__).resolve().parent

# Manually load 'letterbox' from utils/augmentations.py
augment_path = ROOT / 'utils' / 'augmentations.py'
spec = importlib.util.spec_from_file_location("augmentations", augment_path)
augmentations = importlib.util.module_from_spec(spec)
spec.loader.exec_module(augmentations)

# Load trained YOLOv5 model
model_path = ROOT / 'runs' / 'train' / 'fire-detection10' / 'weights' / 'best.pt'
model = torch.load(model_path, map_location='cpu')['model'].float().fuse().eval()

# Input video
video_path = ROOT / 'input.mp4'
cap = cv2.VideoCapture(str(video_path))

# Output video setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(ROOT / 'output_fire.mp4'), fourcc, 20.0, (640, 480))

# Inference settings
img_size = 640
stride = 32

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (640, 480))
    img = augmentations.letterbox(frame_resized, img_size, stride=stride, auto=True)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img)

    img_tensor = torch.from_numpy(img).to('cpu').float()
    img_tensor /= 255.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        pred = model(img_tensor)[0]
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]

    annotated = frame_resized.copy()

    if pred is not None and len(pred):
        pred = pred.detach().cpu().numpy()

        for *box, conf, cls in pred:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(annotated, f'Fire {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow('Fire Detection', annotated)
    out.write(annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()