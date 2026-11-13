import torch
import torch.nn as nn

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)   # 784 -> 128
        self.fc2 = nn.Linear(128, 10)    # 128 -> 10 classes

    def forward(self, x):
        # Flatten ảnh: [batch, 1, 28, 28] -> [batch, 784]
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    #Ảnh 28×28 → flatten thành vector 784.

#fc1: biến 784 → 128 (nén thông tin).

#ReLU: thêm phi tuyến, giữ đặc trưng hữu ích.

#fc2: biến 128 → 10 (kết quả phân loại).

#Output 10 số chính là logits (dự đoán xác suất cho từng lớp 0–9).