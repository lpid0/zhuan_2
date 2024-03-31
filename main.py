import os.path

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.siamese import Siamese
from utils.c import ImageDataset
import argparse

# 创建解析器
parser = argparse.ArgumentParser(description='Process some strings.')

# 添加参数
parser.add_argument('--Epoch', type=int, help='a string to process')
parser.add_argument('--dataset_path', type=str, help='a string to process')

args = parser.parse_args()

if args.Epoch:
    Epoch = args.Epoch
else:
    Epoch = 100
if args.dataset_path:
    dataset_path = args.dataset_path
else:
    dataset_path = "/gemini/data-1"




batch_size = 32
train_dataset = ImageDataset(filepath=dataset_path)
train_loader = DataLoader(train_dataset, batch_size=batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Siamese()
model = model.to(device)

criterion = nn.BCELoss()
optimizer = SGD(model.parameters(), lr=1e-3, momentum=0.9)

num_epochs = Epoch
best_accuracy = 0.0
epochs_no_improve = 0
early_stop = 30

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data1, data2, labels) in enumerate(tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]")):
        data1, data2, labels = data1.to(device), data2.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(data1, data2)

        loss = criterion(outputs.squeeze(), labels.float())
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        predicted = (outputs.squeeze() > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels.float()).sum().item()

    epoch_accuracy = 100 * correct / total
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / (batch_idx + 1)}, Accuracy: {epoch_accuracy}%")

    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")

    torch.save(model.state_dict(), "checkpoints/last.pt")

    if epoch_accuracy > best_accuracy:
        best_accuracy = epoch_accuracy
        torch.save(model.state_dict(), "checkpoints/best.pt")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve == early_stop:
        print(f"No improvement in {early_stop} epochs, stopping training.")
        break
