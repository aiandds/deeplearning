import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)

# Define the data transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load the MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Define the data loader
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the DBN architecture
class DBN(nn.Module):
    def __init__(self):
        super(DBN, self).__init__()
        self.rbm1 = nn.Linear(784, 500)
        self.rbm2 = nn.Linear(500, 200)
        self.rbm3 = nn.Linear(200, 50)
        self.output = nn.Linear(50, 10)

    def forward(self, x):
        h1 = torch.sigmoid(self.rbm1(x))
        h2 = torch.sigmoid(self.rbm2(h1))
        h3 = torch.sigmoid(self.rbm3(h2))
        out = self.output(h3)
        return out

# Instantiate the DBN and define the loss function and optimizer
dbn = DBN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(dbn.parameters(), lr=0.001)

# Train the DBN
num_epochs = 10
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Flatten the images
        images = images.view(-1, 784)
        
        # Convert the data to PyTorch variables
        images = Variable(images)
        labels = Variable(labels)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = dbn(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Print the loss and accuracy every 100 batches
        if (i+1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' %
                  (epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.item()))
            
            # Calculate accuracy on test set
            correct = 0
            total = 0
            for images, labels in test_loader:
                # Flatten the images
                images = images.view(-1, 784)
                
                # Convert the data to PyTorch variables
                images = Variable(images)
                
                # Forward pass
                outputs = dbn(images)
                _, predicted = torch.max(outputs.data, 1)
                
                # Calculate accuracy
                total += labels.size(0)
                correct += (predicted == labels).sum()
            
            print('Accuracy on test set: %d %%' % (100 * correct / total))
