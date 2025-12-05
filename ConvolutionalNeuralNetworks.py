"""Convolutional neural networks have not yet been covered in the course, this will be implemented from scratch,
base on GoodFellow chapter 9 in the reading"""

"""Convolutional neural networks have not yet been covered in the course, this will be implemented from scratch,
base on GoodFellow chapter 9 in the reading"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, numClasses, in_channels=1, baseFilters=3, hiddenFilters=3, dropout=0.1, kernelSize=3, paddingInput=1):
        super().__init__()

        layer1Filters = baseFilters;
        layer2Filters = baseFilters * 2;
        layer3Filters = layer2Filters * 2;
        layer1Dropout = dropout;
        layer2Dropout = dropout * 2;
        layer3Dropout = dropout * 3;

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, layer1Filters, kernel_size=kernelSize, padding=paddingInput),
            nn.BatchNorm2d(layer1Filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(layer1Dropout),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(layer1Filters, layer2Filters, kernel_size=kernelSize, padding=paddingInput),
            nn.BatchNorm2d(layer2Filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(layer2Dropout),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(layer2Filters, layer3Filters, kernel_size=kernelSize, padding=paddingInput),
            nn.BatchNorm2d(layer3Filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(layer3Dropout),
        )

        self.global_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1))

        self.hiddenLayer = nn.Linear(layer3Filters, hiddenFilters)
        self.finalLayer = nn.Linear(hiddenFilters, numClasses)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.hiddenLayer(x))
        x = F.dropout(x, training=self.training)
        x = self.finalLayer(x)

        return x

    def trainOnePass(self, loader, optimizer, critera, device):
        self.train()
        running_loss = 0.0
        running_correct = 0.0
        total = 0;

        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = self(inputs)
            loss = critera(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predictions = torch.max(outputs, 1)
            running_correct += (predictions == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = running_correct / total
        return epoch_loss, epoch_acc

    def predict(self, loader, criterion, device):
        self.eval()
        running_loss = 0.0
        running_correct = 0.0
        total = 0;

        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = self(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predictions = torch.max(outputs, 1)
            running_correct += (predictions == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = running_correct / total
        return epoch_loss, epoch_acc

