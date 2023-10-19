# Flowers 102
import matplotlib.pyplot as plt
import numpy as np

def plot(x, title):
    x_np = x.numpy()
    x_np = x_np.transpose((1, 2, 0))
    x_np = x_np.clip(0, 1)

    fig, ax = plt.subplots()
    if len(x_np.shape) == 2:  # Grayscale
        im = ax.imshow(x_np, cmap='gray')
    else:  # Color
        im = ax.imshow(x_np)
    ax.set_title(title)
    plt.show()
def plot(x,title=None):
    # Move tensor to CPU and convert to numpy
    x_np = x.cpu().numpy()

    # If tensor is in (C, H, W) format, transpose to (H, W, C)
    if x_np.shape[0] == 3 or x_np.shape[0] == 1:
        x_np = x_np.transpose(1, 2, 0)

    # If grayscale, squeeze the color channel
    if x_np.shape[2] == 1:
        x_np = x_np.squeeze(2)

    x_np = x_np.clip(0, 1)

    fig, ax = plt.subplots()
    if len(x_np.shape) == 2:  # Grayscale
        im = ax.imshow(x_np, cmap='gray')
    else:
        im = ax.imshow(x_np)
    plt.title(title)
    ax.axis('off')
    fig.set_size_inches(10, 10)
    plt.show()
# Downloading and extracting the dataset
# Uncomment the following lines if you are running this in a Jupyter Notebook
!wget https://gist.githubusercontent.com/JosephKJ/94c7728ed1a8e0cd87fe6a029769cde1/raw/403325f5110cb0f3099734c5edb9f457539c77e9/Oxford-102_Flower_dataset_labels.txt
!wget https://s3.amazonaws.com/content.udacity-data.com/courses/nd188/flower_data.zip
!unzip 'flower_data.zip'
import torch
from torchvision import datasets, transforms
import os
import pandas as pd

# Directory and transforms
data_dir = '/content/flower_data/'
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

data_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Load the dataset using ImageFolder
dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transform)
dataset_labels = pd.read_csv('Oxford-102_Flower_dataset_labels.txt', header=None)[0].str.replace("'", "").str.strip()

# Load the dataset into a DataLoader for batching
dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
# Extract the batch of images and labels
images, labels = next(iter(dataloader))

print(f"Images tensor shape: {images.shape}")
print(f"Labels tensor shape: {labels.shape}")

i = 100
plot(images[i],dataset_labels[i]);

import torch
from torchvision import models, transforms
import requests
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#define alexnet model
alexnet = models.alexnet(pretrained=True).to(device)
labels = {int(key):value for (key, value) in requests.get('https://s3.amazonaws.com/mlpipes/pytorch-quick-start/labels.json').json().items()}

#transform image for use in model
preprocess = transforms.Compose([
   transforms.Resize(256),
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])
img = images[i]
from torchvision.transforms import ToPILImage
to_pil = ToPILImage()
img = to_pil(img)
img_t = preprocess(img).unsqueeze_(0).to(device)
img_t.shape
# labels
#classify the image with alexnet
scores, class_idx = alexnet(img_t).max(1)
print('Predicted class:', labels[class_idx.item()])
w0 = alexnet.features[0].weight.data
w1 = alexnet.features[3].weight.data
w2 = alexnet.features[6].weight.data
w3 = alexnet.features[8].weight.data
w4 = alexnet.features[10].weight.data
w5 = alexnet.classifier[1].weight.data
w6 = alexnet.classifier[4].weight.data
w7 = alexnet.classifier[6].weight.data
img_t.shape,w0.shape
img_t.shape
img_t[0,:,:,:].shape
def scale(img):
    # Normalize the NumPy array to the range [0, 1]
    max_value = img.max()
    min_value = img.min()
    normalized_array = (img - min_value) / (max_value - min_value)
    return normalized_array
def tensor_plot(img_t,index=0):
    numpy_array = img_t[index,:,:,:].cpu().numpy()
    numpy_array_transposed = numpy_array.transpose(1, 2, 0)
    numpy_array_transposed = scale(numpy_array_transposed)
    plt.imshow(numpy_array_transposed)
    plt.show()
tensor_plot(img_t)
w0.shape
f0 = F.conv2d(img_t, w0, stride=4, padding=2)
f0.shape
i = 0
plt.imshow(f0[0,i,:,:].cpu().numpy())
import torch
import matplotlib.pyplot as plt

# Define a custom function to train and evaluate your model
def train_and_evaluate_model(model, dataloaders, criterion, optimizer, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Track the best accuracy
    best_acc = 0.0

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'Phase: {phase}, Epoch: {epoch}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    model.load_state_dict(best_model_wts)
    return model, best_acc

# Example usage
# Adjust these hyperparameters to achieve a target accuracy of at least 75%
learning_rate = 0.001  # Modify the learning rate
batch_size = 64  # Modify the batch size
num_epochs = 10  # Modify the number of epochs


f0.shape,w0.shape
plot_feature_maps_with_filters(f0, w0)
for phase in ['train', 'val']:
    # Use 'phase' here within the loop
    if phase == 'train':
        # Your code for the training phase
        print("Training phase")
    else:
        # Your code for the validation phase
        print("Validation phase")

import torch
from torchvision import datasets, transforms
import os

# Load the dataset using the validation transform
data_transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'valid'), data_transform_val)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

correct = 0
total = 0

with torch.no_grad():
    alexnet.eval()  # Set the model to evaluation mode
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = alexnet(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Validation accuracy of AlexNet on the Flowers 102 dataset: {accuracy:.2f}%')
