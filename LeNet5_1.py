import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import data
from PIL import Image
import io

#Set Important Variables
batch_size = 64
num_classes = 10
learning_rate = .001 
num_epochs = 10

class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,6, kernel_size=5, stride=1, padding=0),
                nn.BatchNorm2d(6),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Linear(400,120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120,84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)
    
    def forward(self,x):
        layer1_output = self.layer1(x)
        layer2_output = self.layer2(layer1_output)
        layer2_output = layer2_output.reshape(layer2_output.size(0),-1)
        fc_output = self.fc(layer2_output)
        relu_output = self.relu(fc_output)
        fc1_output = self.fc1(relu_output)
        relu1_output = self.relu1(fc1_output)
        result = self.fc2(relu1_output)
        return result

model = LeNet5(num_classes)

#Loss function
cost = nn.CrossEntropyLoss()

#Optimizer
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)

steps = len(data.df_train)

#FUnction to resize data
def resize_image(image_bytes, new_size):  # Adjust the size as needed
    # Convert bytes to an image
    image = Image.open(io.BytesIO(image_bytes["bytes"]))
    # Resize the image
    image = image.resize(new_size)
    # Save the resized image back to bytes
    output = io.BytesIO()
    image.save(output, format='PNG')  # Specify the format
    return output.getvalue()

#Data after being resized to 32x32
#data.df_train['image'] = data.df_train['image'].apply(lambda x: resize_image(x, (32, 32)))  # Example size

preprocess = transforms.Compose([
    transforms.Resize((32, 32)),  # Example size
    transforms.ToTensor(),       # Convert to PyTorch Tensor
])

data.df_train['image'] = data.df_train['image'].apply(preprocess)

total_step = len(data.df_train)
for epoch in range(num_epochs):
    for row in data.df_train.itertuples():  
        #Forward pass
        outputs = model(row.image)
        loss = cost(outputs, row.label)
        #Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 400 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
.format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        
