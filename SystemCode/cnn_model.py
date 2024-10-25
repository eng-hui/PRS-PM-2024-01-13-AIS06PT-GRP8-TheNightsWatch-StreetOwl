import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, num_classes=3):
        super(CNNModel, self).__init__()
        # Define the CNN layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(512 * 16 * 16, 1024)  # Adjust for 512x512 input images
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Forward pass through CNN layers
        x = self.pool(nn.ReLU()(self.conv1(x)))  # Conv1 + ReLU + Pooling
        x = self.pool(nn.ReLU()(self.conv2(x)))  # Conv2 + ReLU + Pooling
        x = self.pool(nn.ReLU()(self.conv3(x)))  # Conv3 + ReLU + Pooling
        x = self.pool(nn.ReLU()(self.conv4(x)))  # Conv4 + ReLU + Pooling
        x = self.pool(nn.ReLU()(self.conv5(x)))  # Conv5 + ReLU + Pooling
        x = x.view(-1, 512 * 16 * 16)  # Flatten the output
        x = nn.ReLU()(self.fc1(x))  # Fully connected layer 1 + ReLU
        x = self.dropout(x)  # Dropout for regularization
        x = self.fc2(x)  # Output layer
        return x

    def predict(self, inputs):
        self.eval()  
        with torch.no_grad():  
            outputs = self.forward(inputs)
            _, preds = torch.max(outputs, 1)  
        return preds
