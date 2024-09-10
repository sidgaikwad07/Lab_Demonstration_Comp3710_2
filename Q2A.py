import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import numpy as np

# Load dataset
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
X = lfw_people.images
Y = lfw_people.target
# Define the number of classes
num_classes = len(np.unique(Y))

# Verify the value range of X
print("X_min:", X.min(), "X_max:", X.max())

# Split into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

# Reshape data to fit CNN input [batch_size, channels, height, width]
X_train = X_train[:, np.newaxis, :, :]
X_test = X_test[:, np.newaxis, :, :]
print("X_train shape:", X_train.shape)  # Should output [n_samples, 1, h, w]

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

class CNNClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)  # Increased filters from 32 to 64
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Increased filters from 32 to 128
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.5)  # Added dropout with 50% probability
        
        # Calculate the size of the flattened feature map dynamically
        self._to_linear = None
        self.convs(torch.randn(1, 1, X_train.shape[2], X_train.shape[3]))  # Pass a dummy input with the same dimensions as the images
        
        self.fc1 = nn.Linear(self._to_linear, 256)  # Increased to 256 units
        self.fc2 = nn.Linear(256, num_classes)
    
    def convs(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        if self._to_linear is None:
            self._to_linear = x[0].numel()  # Flatten the tensor and get the number of elements
        
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)
        return x

# Instantiate the model
model = CNNClassifier(num_classes)

# Hyperparameters
learning_rate = 0.0001  # Reduced learning rate
batch_size = 32         # Mini-batch size
num_epochs = 50         # Increased number of epochs

# DataLoader for mini-batch training
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop with mini-batch processing
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
    print(f'Accuracy: {accuracy:.4f}')


