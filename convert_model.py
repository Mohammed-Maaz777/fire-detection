import torch
from pathlib import Path
import sys

# Add YOLOv5 root to PATH
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] / 'yolov5'  # yolov5 directory
sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.general import check_img_size

# Config
weights = str(ROOT / 'weights' / 'best.pt')  # path to your best.pt
imgsz = 640
device = 'cpu'

# Load model using DetectMultiBackend (latest method)
model = DetectMultiBackend(weights, device=device)
model.model.float().eval()

# Force export mode to avoid tracing issues
for m in model.model.modules():
    if hasattr(m, 'export'):
        m.export = True

# Dummy input
dummy_input = torch.zeros(1, 3, imgsz, imgsz).to(device)

# Export TorchScript
traced = torch.jit.trace(model.model, dummy_input)
traced.save(str(ROOT / 'weights' / 'best.torchscript.pt'))
print("✅ TorchScript model saved")

# Optional: Export ONNX
try:
    import onnx
    torch.onnx.export(model.model, dummy_input, str(ROOT / 'weights' / 'best.onnx'),
                      input_names=['images'], output_names=['output'],
                      opset_version=12)
    print("✅ ONNX model saved")
except ImportError:
    print("❌ ONNX export failed: install onnx using pip install onnx")
