# Step 1 — Install dependencies
# pip install torch torchvision matplotlib

# Step 2 — Data pipeline (loading + preprocessing)
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
])

train_dataset = ImageFolder(root="cat-vs-dogs-dataset/train", transform=transform)
val_dataset = ImageFolder(root="cat-vs-dogs-dataset/test", transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Step 3 — Building the CNN
import torch.nn as nn
import torch.nn.functional as F

class CatsDogsCNN(nn.Module):
    def __init__(self):
        super(CatsDogsCNN, self).__init__()

        # Convolution blocks
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)

        self.pool = nn.MaxPool2d(2,2)

        # Fully connected layers
        self.fc1 = nn.Linear(128*8*8, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512,1)

    def forward(self,x):

        x = self.pool(F.relu(self.conv1(x)))   # 128 -> 64
        x = self.pool(F.relu(self.conv2(x)))   # 64 -> 32
        x = self.pool(F.relu(self.conv3(x)))   # 32 -> 16
        x = self.pool(F.relu(self.conv4(x)))   # 16 -> 8

        x = x.view(-1, 128*8*8)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = torch.sigmoid(self.fc2(x))

        return x
    
model = CatsDogsCNN()

# Step 4 — Loss function and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Step 5 — Training loop
num_epochs = 10

for epoch in range(num_epochs):

    model.train()
    running_loss = 0

    for images, labels in train_loader:

        labels = labels.float().unsqueeze(1)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

# Step 6 — Evaluation
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:

        labels = labels.float().unsqueeze(1)

        outputs = model(images)
        predicted = (outputs > 0.5).float()

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print("Validation Accuracy:", accuracy)

# RESULT
# performed pretty poorly I think
# Epoch 1, Loss: 0.6975880761941274
# Epoch 2, Loss: 0.6931971146000756
# Epoch 3, Loss: 0.6919169392850664
# Epoch 4, Loss: 0.6935399803850386
# Epoch 5, Loss: 0.6805553601847755
# Epoch 6, Loss: 0.6784445080492232
# Epoch 7, Loss: 0.6658895015716553
# Epoch 8, Loss: 0.6703467998239729
# Epoch 9, Loss: 0.6697428193357255
# Epoch 10, Loss: 0.6534032622973124

# Validation Accuracy: 56.42857142857143
