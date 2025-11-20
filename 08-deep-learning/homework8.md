### Dataset Preparation

First, let's download and prepare the dataset:

```python
import requests
import zipfile
import os
import io

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Download the dataset
print("Downloading dataset...")
url = "https://github.com/SVizor42/ML_Zoomcamp/releases/download/straight-curly-data/data.zip"
response = requests.get(url)
print("Download complete!")

# Extract the dataset
print("Extracting dataset...")
with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
    zip_ref.extractall(".")
print("Extraction complete!")
```

Let's install PyTorch and other required libraries:

```python
# Run this cell to install the required packages
!pip install torch torchvision torchaudio
```

Let's set up the reproducibility settings:

```python
import numpy as np
import torch

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Now, let's import the necessary libraries and check the dataset structure:

```python
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Check the dataset structure
print("Training data:")
print(os.listdir("data/train"))
print("\nTest data:")
print(os.listdir("data/test"))
```

### Model Definition

Let's define the CNN model according to the specifications:

```python
class HairClassifier(nn.Module):
    def __init__(self):
        super(HairClassifier, self).__init__()
        
        # Convolutional layer with 32 filters, kernel size 3x3, and ReLU activation
        self.conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3))
        self.relu = nn.ReLU()
        
        # Max pooling layer with pool size 2x2
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        
        # Calculate the size after convolution and pooling
        # Input: (3, 200, 200)
        # After Conv2d(3, 32, 3x3): (32, 198, 198)
        # After MaxPool2d(2x2): (32, 99, 99)
        self.flatten_size = 32 * 99 * 99
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 64)
        self.fc2 = nn.Linear(64, 1)
        
        # No need to define sigmoid here as we'll use BCEWithLogitsLoss
        # which combines sigmoid and BCE in one function
        
    def forward(self, x):
        # Convolutional layer
        x = self.conv(x)
        x = self.relu(x)
        
        # Max pooling
        x = self.pool(x)
        
        # Flatten
        x = x.view(-1, self.flatten_size)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

# Create the model
model = HairClassifier().to(device)

# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.8)

# Print model summary
print(model)
```

## Question 1: Loss Function

For binary classification, we need to choose an appropriate loss function. Since our model outputs a single value without a sigmoid activation (as per the instructions), and we're doing binary classification, the appropriate loss function is:

```python
criterion = nn.BCEWithLogitsLoss()
```

BCEWithLogitsLoss combines a sigmoid activation and binary cross-entropy loss in one function, which is numerically more stable than using a separate sigmoid followed by BCELoss.


## Question 2: Total Number of Parameters

Let's count the total number of parameters in our model:

```python
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
```

Number of parameters `20 073 473`.


### Data Loaders

Now, let's set up the data loaders with the specified transformations:

```python
# Define transformations
train_transforms = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ) # ImageNet normalization
])

test_transforms = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ) # ImageNet normalization
])

# Load datasets
train_dataset = datasets.ImageFolder(root='data/train', transform=train_transforms)
test_dataset = datasets.ImageFolder(root='data/test', transform=test_transforms)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)

# Print class mappings
print(f"Class mappings: {train_dataset.class_to_idx}")
```

### Training the Model

Let's train the model for 10 epochs:

```python
num_epochs = 10
history = {'acc': [], 'loss': [], 'val_acc': [], 'val_loss': []}

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        labels = labels.float().unsqueeze(1) # Ensure labels are float and have shape (batch_size, 1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        # For binary classification with BCEWithLogitsLoss, apply sigmoid to outputs before thresholding for accuracy
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = correct_train / total_train
    history['loss'].append(epoch_loss)
    history['acc'].append(epoch_acc)

    model.eval()
    val_running_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.float().unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * images.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_epoch_loss = val_running_loss / len(test_dataset)
    val_epoch_acc = correct_val / total_val
    history['val_loss'].append(val_epoch_loss)
    history['val_acc'].append(val_epoch_acc)

    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, "
          f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}")
```

## Question 3: Median of Training Accuracy

Let's calculate the median of training accuracy for all epochs:

```python
import numpy as np
# Calculate median of training accuracy
median_train_acc = np.median(history['acc'])
print(f"Median training accuracy: {median_train_acc:.4f}")
```

Median training accuracy is `0.8175`.

## Question 4: Standard Deviation of Training Loss

Let's calculate the standard deviation of training loss for all epochs:

```python
# Calculate standard deviation of training loss
std_train_loss = np.std(history['loss'])
print(f"Standard deviation of training loss: {std_train_loss:.4f}")
```

Standard deviation of training loss is `0.1590`.

### Data Augmentation

Now, let's add data augmentations to the training transforms:

```python
# Define transformations with augmentations
train_transforms_aug = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.RandomRotation(50),
    transforms.RandomResizedCrop(200, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ) # ImageNet normalization
])

# Load dataset with augmentations
train_dataset_aug = datasets.ImageFolder(root='data/train', transform=train_transforms_aug)

# Create data loader with augmentations
train_loader_aug = DataLoader(train_dataset_aug, batch_size=20, shuffle=True)
```

### Training with Augmentations

Let's continue training the model for 10 more epochs with the augmented data:

```python
num_epochs = 10
history_aug = {'acc': [], 'loss': [], 'val_acc': [], 'val_loss': []}

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for images, labels in train_loader_aug:
        images, labels = images.to(device), labels.to(device)
        labels = labels.float().unsqueeze(1) # Ensure labels are float and have shape (batch_size, 1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        # For binary classification with BCEWithLogitsLoss, apply sigmoid to outputs before thresholding for accuracy
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_dataset_aug)
    epoch_acc = correct_train / total_train
    history_aug['loss'].append(epoch_loss)
    history_aug['acc'].append(epoch_acc)

    model.eval()
    val_running_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.float().unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * images.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_epoch_loss = val_running_loss / len(test_dataset)
    val_epoch_acc = correct_val / total_val
    history_aug['val_loss'].append(val_epoch_loss)
    history_aug['val_acc'].append(val_epoch_acc)

    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, "
          f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}")
```

## Question 5: Mean of Test Loss with Augmentations

Let's calculate the mean of test loss for all epochs with augmentations:

```python
# Calculate mean of test loss with augmentations
mean_val_loss_aug = np.mean(history_aug['val_loss'])
print(f"Mean test loss with augmentations: {mean_val_loss_aug:.4f}")
```

Mean test loss with augmentations is `0.5465`.

## Question 6: Average Test Accuracy for Last 5 Epochs with Augmentations

Let's calculate the average test accuracy for the last 5 epochs with augmentations:

```python
# Calculate average test accuracy for the last 5 epochs with augmentations
avg_val_acc_last_5 = np.mean(history_aug['val_acc'][5:])
print(f"Average test accuracy for last 5 epochs with augmentations: {avg_val_acc_last_5:.4f}")
```

Average test accuracy for last 5 epochs with augmentations is `0.7353`.