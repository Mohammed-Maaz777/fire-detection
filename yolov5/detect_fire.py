import sys
import argparse
import cv2
import torch
import numpy as np
import importlib.util
from pathlib import Path
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

# Inference settings
img_size = 640
stride = 32

def detect_fire(video_path, output_path):
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    # Get frame size from input
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (width, height))
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

        out.write(annotated)

    cap.release()
    out.release()
    print(f"Saved fire detection result to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='Path to input video')
    parser.add_argument('--output', type=str, required=True, help='Path to save output video')
    args = parser.parse_args()

    detect_fire(args.source, args.output)
