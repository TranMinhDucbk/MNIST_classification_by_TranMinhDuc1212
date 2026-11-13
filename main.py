import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from mnist_model import MNISTModel

# -----------------
# 1. Dataloader
# -----------------
transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.5,), (0.5,))
])
#img = ToTensor()(img)
#img = Normalize((0.5,), (0.5,))(img)

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset  = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# -----------------
# 2. Model, Loss, Optimizer
# -----------------
device = torch.device ("cpu")
model = MNISTModel().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------
# 3. Training loop
# -----------------
epochs = 5
for epoch in range(epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device) #đưa lên cpu

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# -----------------
# 4. Testing
# -----------------
model.eval()
#correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        val_loss=criterion(outputs, labels)
        print(f"Val loss: {val_loss:.4f}")
        #_, predicted = torch.max(outputs.data, 1) #.data cắt tensor ra khổi autograd
        #total += labels.size(0)
       # correct += (predicted == labels).sum().item()


#print(f"Accuracy on test set: {100 * correct / total:.2f}%")

os.makedirs('models', exist_ok=True)#kiểm tra thư mục tồn tại chưa       #tạo thư mục
save_path = os.path.join('models', 'mnist_model.pth') 
torch.save(model.state_dict(), save_path)  
print(f"Model weights saved to {save_path}")
