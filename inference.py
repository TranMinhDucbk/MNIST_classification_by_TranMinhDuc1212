import numpy as np
import onnxruntime as ort
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

# -------------------------
# 1. Preprocessing
# -------------------------
# Step 1: Data cleaning  (ở MNIST thì dữ liệu đã sạch, không cần xóa missing values)
# Step 2: Data integration (không cần vì dữ liệu đã chuẩn sẵn từ torchvision)
# Step 3: Data transformation (chuẩn hóa ảnh, scale pixel)
# Step 4: Data reduction (chọn batch nhỏ thay vì full dataset)

transform = transforms.Compose([
    transforms.ToTensor(),                  # chuyển ảnh thành tensor [0,1]
   # transforms.Normalize((0.5,), (0.5,))    # chuẩn hóa về mean=0.5, std=0.5
])

test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=True)

# Lấy 1 ảnh từ test set
images, labels = next(iter(test_loader))   # images: [1,1,28,28]
input_tensor = images.numpy()              # convert sang numpy cho ONNX


# -------------------------
# 2. Executing
# -------------------------
# Step 1: Load model
onnx_model_path = "models/mnist_model.onnx"
session = ort.InferenceSession(onnx_model_path)

# Step 2: Chuẩn bị input/output
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Step 3: Thực thi model
outputs = session.run([output_name], {input_name: input_tensor})
guess  = outputs[0]  


# -------------------------
# 3. Postprocessing
# -------------------------
# Step 1: Chuyển logits thành xác suất
probs = torch.softmax(torch.tensor(guess), dim=1)

# Step 2: Lấy class có xác suất cao nhất
pred_class = torch.argmax(probs, dim=1).item()

# Step 3: So sánh với ground truth
print(f"Ground truth label: {labels.item()}")
print(f"Predicted class: {pred_class}")

