import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from data import splits, df_train, df_test
from PIL import Image
import io
import sys
import os

# Constants
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.0005

class C3ConnectionTable:
    def __init__(self):
        # Table 1 from the paper
        self.table = torch.tensor([
            [1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1],  # 0
            [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1],  # 1
            [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1],  # 2
            [0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1],  # 3
            [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1],  # 4
            [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1]   # 5
        ], dtype=torch.float32)

class RBFLayer(nn.Module):
    def __init__(self, in_features=84, num_classes=10):
        super(RBFLayer, self).__init__()
        self.centers = nn.Parameter(self.generate_centers(in_features, num_classes))
        
    def generate_centers(self, in_features, num_classes):
        # Generate 7x12 bitmap for digits 0-9
        centers = torch.zeros(num_classes, in_features)
        for digit in range(num_classes):
            # Create a 7x12 bitmap representation for each digit
            bitmap = self.create_digit_bitmap(digit)
            # Convert to vector and store
            centers[digit] = bitmap.view(-1)
        return centers
    
    def create_digit_bitmap(self, digit):
        # Simple bitmap representations - this should be enhanced based on DIGIT data
        bitmap = torch.zeros(7, 12)
        if digit == 0:
            bitmap[1:6, 1:11] = 1  # Simple circle
        elif digit == 1:
            bitmap[:, 6] = 1  # Vertical line
        # Add patterns for other digits...
        return bitmap
    
    def forward(self, x):
        # Compute Euclidean distances
        diff = x.unsqueeze(1) - self.centers.unsqueeze(0)
        distances = torch.sum(diff * diff, dim=2)
        return distances


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        #data preprocess block
        # Layers as per the paper
        self.c1 = nn.Conv2d(1, 6, kernel_size=5)

        # MaxPooling instead of Average Pooling
        self.s2 = nn.MaxPool2d(kernel_size=2, stride=2)  

        #self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(6, 16, kernel_size=5)
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c5 = nn.Conv2d(16, 120, kernel_size=5)
        self.f6 = nn.Linear(120, 84)

        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.5)  
         # Softmax activation for classification
        self.softmax = nn.Softmax(dim=1) 
        
        self.output = RBFLayer(84, 10)
        self.c3_connections = C3ConnectionTable()
        
    def activation(self, x):
        return 1.7159 * torch.tanh(2/3 * x)
    
    def forward(self, x):
        x = self.activation(self.c1(x))
        x = self.s2(x)
        x = self.activation(self.c3(x))
        x = self.s4(x)
        x = self.activation(self.c5(x))
        x = x.view(-1, 120)
        x = self.activation(self.f6(x))
        x = self.output(x)
        return x

def discriminative_loss(outputs, targets, j=0.1):
    batch_size = outputs.size(0)
    correct_penalties = outputs[range(batch_size), targets]
    exp_terms = torch.exp(-outputs)
    exp_correct = torch.exp(-correct_penalties)
    loss = torch.mean(correct_penalties + torch.log(j + torch.sum(exp_terms, dim=1)))
    return loss

# class SDLMOptimizer:
#     def __init__(self, parameters, eta=0.0005, mu=0.02):
#         self.parameters = list(parameters)
#         self.eta = eta
#         self.mu = mu
#         self.hkk = {}
    
#     def zero_grad(self):
#         for param in self.parameters:
#             if param.grad is not None:
#                 param.grad.zero_()
        
#     def compute_hkk(self, loss, param):
#         grad = torch.autograd.grad(loss, param, create_graph=True, retain_graph=True)[0]
#         hkk = torch.zeros_like(param)
#         for i in range(param.numel()):
#             second_deriv = torch.autograd.grad(grad.view(-1)[i], param, retain_graph=True)[0]
#             hkk.view(-1)[i] = second_deriv.view(-1)[i]
#         return hkk
    
#     def step(self, loss):
#         with torch.enable_grad():
#             for param in self.parameters:
#                 if param.grad is None:
#                     continue
#                 hkk = self.compute_hkk(loss, param)
#                 step_size = self.eta / (self.mu + torch.abs(hkk))  # Added abs to prevent division by negative numbers
#                 param.data.add_(-step_size * param.grad.data)

class SDLMOptimizer:
    def __init__(self, parameters, eta=0.0005, mu=0.02):
        self.parameters = list(parameters)
        self.eta = eta
        self.mu = mu
        
    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.grad.zero_()
    
    def step(self, loss):
        # Compute gradients using backward
        loss.backward(retain_graph=True)
        
        with torch.no_grad():
            for param in self.parameters:
                if param.grad is None:
                    continue
                    
                # Compute second derivatives
                grad = param.grad
                hkk = torch.zeros_like(param.data)
                
                # Update parameters using SDLM update rule
                step_size = self.eta / (self.mu + torch.abs(hkk) + 1e-8)  # Added small constant for numerical stability
                param.data.add_(-step_size * grad)
        
        # Zero gradients after update
        self.zero_grad()
    
# Data preprocessing
def preprocess_data(df):
    # Convert images to tensors and resize to 32x32
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    images = []
    labels = []
    
    for row in df.itertuples():
        img = Image.open(io.BytesIO(row.image["bytes"]))
        img_tensor = transform(img)
        images.append(img_tensor)
        labels.append(row.label)
    
    return torch.stack(images), torch.tensor(labels)

# Training utilities
def compute_accuracy(outputs, targets):
    _, predicted = torch.max(-outputs.data, 1)
    return (predicted == targets).sum().item() / targets.size(0)

def update_confusion_matrix(outputs, targets, confusion_matrix):
    _, predicted = torch.max(-outputs.data, 1)
    for t, p in zip(targets.view(-1), predicted.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1
    return confusion_matrix


def train_and_evaluate():
    # Prepare data
    train_images, train_labels = preprocess_data(df_train)
    test_images, test_labels = preprocess_data(df_test)
    
    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model and optimizer
    model = LeNet5()
    optimizer = SDLMOptimizer(model.parameters())
    
    # Training history
    train_errors = []
    test_errors = []
    confusion_matrix = torch.zeros(10, 10)
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_error = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(data)
            loss = discriminative_loss(outputs, target)
            
            # Step the optimizer
            optimizer.step(loss)
            
            # Compute accuracy
            with torch.no_grad():
                train_error += 1 - compute_accuracy(outputs, target)
            
        train_error /= len(train_loader)
        train_errors.append(train_error)
        
        # Evaluation
        model.eval()
        test_error = 0
        confusion_matrix.zero_()
        
        with torch.no_grad():
            for data, target in test_loader:
                outputs = model(data)
                test_error += 1 - compute_accuracy(outputs, target)
                confusion_matrix = update_confusion_matrix(outputs, target, confusion_matrix)
        
        test_error /= len(test_loader)
        test_errors.append(test_error)
        
        print(f'Epoch {epoch+1}: Train Error: {train_error:.4f}, Test Error: {test_error:.4f}')
    
    return model, train_errors, test_errors, confusion_matrix

# Visualization functions
def plot_error_rates(train_errors, test_errors):
    plt.figure(figsize=(10, 6))
    plt.plot(train_errors, label='Training Error')
    plt.plot(test_errors, label='Test Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error Rate')
    plt.title('Training and Test Error Rates')
    plt.legend()
    plt.show()

def plot_confusion_matrix(confusion_matrix):
    plt.figure(figsize=(10, 10))
    plt.imshow(confusion_matrix.numpy(), cmap='Blues')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def find_most_confusing_examples(model, test_loader):
    model.eval()
    most_confusing = {i: {'confidence': 0, 'image': None, 'true_label': None} for i in range(10)}
    
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            confidences, predictions = torch.max(-outputs.data, 1)
            
            for i, (pred, true, conf, img) in enumerate(zip(predictions, target, confidences, data)):
                if pred != true and conf > most_confusing[pred.item()]['confidence']:
                    most_confusing[pred.item()] = {
                        'confidence': conf.item(),
                        'image': img.numpy(),
                        'true_label': true.item()
                    }
    
    return most_confusing

# Main execution
if __name__ == '__main__':
    model, train_errors, test_errors, confusion_matrix = train_and_evaluate()
    
    # Plot results
    plot_error_rates(train_errors, test_errors)
    plot_confusion_matrix(confusion_matrix)
    
    test_images, test_labels = preprocess_data(df_test)
    # Find most confusing examples
    test_loader = DataLoader(TensorDataset(test_images, test_labels), batch_size=BATCH_SIZE)
    confusing_examples = find_most_confusing_examples(model, test_loader)
    
    # Display most confusing examples
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axes.flat):
        example = confusing_examples[i]
        ax.imshow(example['image'][0], cmap='gray')
        ax.set_title(f'True: {example["true_label"]}\nPred: {i}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()
