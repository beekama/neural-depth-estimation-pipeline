import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
import argparse
import UNet
import UNetResNet
import UNetRplus

BATCH_SIZE = 1
EPOCHES=10
COMBINED = False

model_choices = {
    'Unet': UNet,
    'Unetresnet': UNetResNet,
    'Unetplus': UNetRplus,
    }

def save_img(image, path):
    # float32 to uint8
    img = np.clip((image * 255), 0, 255).astype(np.uint8)
    img = transforms.ToPILImage()(img)
    img.save(path)

def train(model, device, train_loader, valid_loader, criterion, optimizer, num_epoches):

    model.to(device)

    for epoch in range(num_epoches):
        model.train()
        train_loss = 0.0
        for images, depths in train_loader:
            images = images.to(device)
            depths = depths.to(device)

            # reset gradients of all tensors to zero
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, depths)  
            # calculate gradients with respect to loss
            loss.backward()
            # update model parameters
            optimizer.step()

            train_loss += loss.item()
    
        model.eval()
        valid_loss = 0.0
        for images, depths in valid_loader:
            images = images.to(device)
            depths = depths.to(device)

            outputs = model(images)
            loss = criterion(outputs, depths)
            valid_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epoches} - Loss: {train_loss / len(train_loader):.4f} \t\t Validation Loss: {valid_loss / len(valid_loader)}")

    torch.save(model.state_dict(), "model.pth")

def test(model, device, test_loader, criterion):
    # enable evaluation mode
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    model.to(device)
    test_loss = 0.0
    results = []

    with torch.no_grad():
        for images, depths in test_loader:
            images = images.to(device)
            depths = depths.to(device)

            outputs = model(images)
            loss = criterion(outputs, depths) 
            
            test_loss += loss.item()

            # store image, estimated depth and true depth for plotting
            batch_results = {
                "images": images.cpu(),
                "depth_gt": depths.cpu(),
                "depth_pred": outputs.cpu()
            }
            results.append(batch_results)
    return results


    print(f"Test Loss: {test_loss / len(test_loader):.4f}")

def plot(results, output_dir, model_type, combined):
    # Create directory to save images
    if model_type == UNet:
        result_dir = output_dir + "/depth_results_unet"
    elif model_type == UNetResNet:
        result_dir = output_dir + "/depth_results_unetresnet"
    elif model_type == UNetRplus:
        result_dir = output_dir + "/depth_results_unetrplus"
    else:
        raise Exception(Exception("Unknown Model - unable to set output-path"))
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    for i, batch_results in enumerate(results):
        if combined:
            images = batch_results["images"].permute(0,2,3,1)[:,:,:,:3].numpy()
        else:
            images = batch_results["images"].permute(0,2,3,1).numpy()
        depth_gt = batch_results["depth_gt"].squeeze(1).numpy()
        depth_pred = batch_results["depth_pred"].squeeze(1).numpy()


        for j in range(images.shape[0]):
            image_path = f"{result_dir}/{(i*BATCH_SIZE+j):03d}_image.png"
            detph_gt_path = f"{result_dir}/{(i*BATCH_SIZE+j):03d}_depth_gt.png"
            depth_pred_path = f"{result_dir}/{(i*BATCH_SIZE+j):03d}_depth_pred.png"

            save_img(images[j], image_path)
            save_img(depth_gt[j], detph_gt_path)
            save_img(depth_pred[j], depth_pred_path)

class DepthDataset(Dataset):
    def __init__(self, data_path, transform=None, combined=False):
        self.data_path = data_path
        self.image_files = os.listdir(data_path)
        self.transform = transform
        self.combined = combined

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image_path = os.path.join(self.data_path, image_file)
        depth_path = os.path.join(self.data_path, "../depth_maps", image_file)

        if self.combined:
            image = np.loadtxt(image_path).reshape( 512, 512, 6)
        else:
            image = Image.open(image_path).convert("RGB")
        depth = Image.open(depth_path).convert("L")

        if self.transform is not None:
            image = self.transform(image)
            depth = self.transform(depth)

        return image.float(), depth     ## todo evtl float fuer normale bilder weg



#############################
### Set up Neural Network ###
#############################
def depthestimation(output_dir, training, num_epoches, model_type, combined):
    # resolv model_type
    model_type = model_choices[model_type]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config={'in_channels': 3, 'out_channels': 1, 'features': [64, 128, 256, 512]}
    model = model_type.Model(config)

    # Transformation-function for images 
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Set up loaders for training and test data
    train_dataset = DepthDataset(output_dir + "/train", transform=transform, combined=combined)
    valid_dataset = DepthDataset(output_dir + "/valid", transform=transform, combined=combined)
    test_dataset = DepthDataset(output_dir + "/test", transform=transform, combined=combined)

    # todo: adapt batch_size - 1
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Define the loss function and optimizer 
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if training:
        train(model, device, train_loader, valid_loader, criterion, optimizer, num_epoches)
    results = test(model, device, test_loader, criterion)
    plot(results, output_dir, model_type, combined)

if __name__ == "__main__":
    

    parser = argparse.ArgumentParser(description='neuronal depth estimator',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    

    parser.add_argument("--helpme", "-help", action="help", help="Show the helper")
    parser.add_argument('--folder', '-f', help='folder, which contains test, train and depthfolders', required=True)
    parser.add_argument("--num_epoches", "-epoches", help="num of training-epoches", default=EPOCHES)
    parser.add_argument('--model', choices={'Unet', 'Unetresnet', 'Unetplus'}, help="select model type", required=True)
    parser.add_argument('--combined', action='store_true', help='Process combined input-image (eg. infrared & normal)')

    args = parser.parse_args()

    depthestimation(args.folder, True, args.num_epoches, args.model, args.combined)