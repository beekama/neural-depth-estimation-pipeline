import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
import argparse

class DepthEstimationNet(nn.Module):
    def __init__(self):
        super(DepthEstimationNet, self).__init__()

        ### network layers
        # 3-channel-input(RGB), apply 64 filters of size 3x3
        # move filter one pixel at a time (stride)
        # add one pixel at each side to keep spatial dimension
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

        # Output a single-channel depth map
        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)  


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)

        return x


class DepthDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.image_files = os.listdir(data_path)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image_path = os.path.join(self.data_path, image_file)
        depth_path = os.path.join(self.data_path, "../depth_maps", image_file)

        # todo: maybe ajust size
        image = Image.open(image_path).convert("RGB").resize((100, 190))
        depth = Image.open(depth_path).convert("L").resize((100, 190))

        if self.transform is not None:
            image = self.transform(image)
            depth = self.transform(depth)

        return image, depth



#############################
### Set up Neural Network ###
#############################
parser = argparse.ArgumentParser(description='neuronal depth estimator',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--helpme", "-help", action="help", help="Show the helper")

parser.add_argument('--folder', '-f', help='folder, which contains test, train and depthfolders', required=True)

args = parser.parse_args()

data_path = args.folder
model = DepthEstimationNet()

# Transformation-function for images 
transform = transforms.Compose([
    transforms.ToTensor()
])

# Set up loaders for training and test data
train_dataset = DepthDataset(data_path + "/train", transform=transform)
test_dataset = DepthDataset(data_path + "/test", transform=transform)
#train_size = int(0.8 * len(dataset))
#test_size = len(dataset) - train_size
#train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# todo: adapt batch_size - 1
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


#####################
### TRAINING LOOP ###
#####################
num_epochs = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    current_loss = 0.0

    for images, depths in train_loader:
        images = images.to(device)
        depths = depths.to(device)

        # reset gradients of all tensors to zero
        optimizer.zero_grad()

        outputs = model(images)
        # add new dimension for single-channel depth map
        loss = criterion(outputs, depths.unsqueeze(1))  
        # calculate gradients with respect to loss
        loss.backward()
        # update model parameters
        optimizer.step()

        current_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {current_loss / len(train_loader):.4f}")


####################
### TESTING LOOP ###
####################

# enable evaluation mode
model.eval()
test_loss = 0.0
results = []

with torch.no_grad():
    for images, depths in test_loader:
        images = images.to(device)
        depths = depths.to(device)

        outputs = model(images)
        # add new dimension for single-channel depth map
        loss = criterion(outputs, depths.unsqueeze(1)) 
        
        test_loss += loss.item()

        # store image, estimated depth and true depth for plotting
        batch_results = {
            "images": images.cpu(),
            "depth_gt": depths.cpu(),
            "depth_pred": outputs.cpu()
        }
        results.append(batch_results)


    print(f"Test Loss: {test_loss / len(test_loader):.4f}")
'''
# Plotting results
num_samples = min(5, len(results))
print(len(results))
fig, axes = plt.subplots(num_samples, 3, figsize=(256,256))

for i in range(num_samples):
    image = np.transpose(results[i]["images"][0], (1, 2, 0))
    depth_gt = np.squeeze(results[i]["depth_gt"][0])
    depth_pred = np.squeeze(results[i]["depth_pred"][0])

    axes[i].imshow(image)
    axes[i].set_title("Image")
    axes[i].axis("off")

    axes[1].imshow(depth_gt, cmap="gray")
    axes[1].set_title("Ground Truth Depth")
    axes[1].axis("off")

    axes[2].imshow(depth_pred, cmap="gray")
    axes[2].set_title("Predicted Depth")
    axes[2].axis("off")

plt.tight_layout()
plt.show()
'''