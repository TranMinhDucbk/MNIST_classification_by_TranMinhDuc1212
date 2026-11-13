import torch
from mnist_model import MNISTModel
import onnxruntime

model = MNISTModel()
# load pretrained weight (đã train từ main.py)
model.load_state_dict(torch.load("models/mnist_model.pth", map_location="cpu"))
model.eval()  # chuyển sang chế độ inference

dummy_input = torch.randn(1, 1, 28, 28)  # batch=1, channel=1, size=28x28

# Đường dẫn lưu file .onnx
save_path = "models/mnist_model.onnx"

# Xuất mô hình
torch.onnx.export(
    model,                 # mô hình đã load weight
    dummy_input,           # dummy input (giả lập ảnh đầu vào)
    save_path,             # nơi lưu file onnx
    input_names=["input"], # tên của input tensor
    output_names=["output"], # tên của output tensor
    dynamic_axes={         # cho phép batch_size linh động
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    },
    export_params=True,    # có lưu pretrained weight vào file onnx
    verbose=True           # in ra đồ thị mô hình ở dạng text
)

print(f" Model converted to ONNX and saved at {save_path}")