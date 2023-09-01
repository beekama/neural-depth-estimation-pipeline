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
import oniroResnet
sys.path.append('SMDE/models/networks')
import monodepth2

from torchvision.utils import save_image

BATCH_SIZE = 1
EPOCHES=10

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
            print("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT")
            print(outputs)
            count = 0
            for el in outputs:
                im = el
                save_image(im, "bild" + str(count) + ".png")
                count+=1
            outputs_gray = torch.mean(outputs[0], dim=1, keepdim=True)
            loss = criterion(outputs_gray, depths)  
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
            print("VAAAAAAAAAAAAAAAAAAAAAAALLLLLL")
            print(outputs)
            for el in outputs:
                print(el)
            outputs_gray = torch.mean(outputs[0], dim=1, keepdim=True)
            loss = criterion(outputs_gray, depths)
            valid_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epoches} - Loss: {train_loss / len(train_loader):.4f} \t\t Validation Loss: {valid_loss / len(valid_loader)}")

    torch.save(model.state_dict(), "model.pth")

def test(model, device, test_loader, criterion):
    # enable evaluation mode
    #config={'in_channels': 3, 'out_channels': 1, 'features': [64, 128, 256, 512]}
    #model = resnet.ResNet(resnet.Bl,[64, 128, 256, 512], num_classes=1)
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
            outputs_gray = torch.mean(outputs[0], dim=1, keepdim=True)
            loss = criterion(outputs_gray, depths) 
            
            test_loss += loss.item()
            print("DEOHT: " + str(images.cpu().shape))
            print("DEOHT: " + str(outputs_gray.cpu().shape))
            # store image, estimated depth and true depth for plotting
            batch_results = {
                "images": images.cpu(),
                "depth_gt": depths.cpu(),
                "depth_pred": outputs_gray.cpu()
            }
            results.append(batch_results)
    return results


    print(f"Test Loss: {test_loss / len(test_loader):.4f}")

def plot(results, output_dir):
    # Create directory to save images
    result_dir = output_dir + "/depth_results"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    for i, batch_results in enumerate(results):
    #for i in range(num_samples):
        images = batch_results["images"].permute(0,2,3,1).numpy()
        depth_gt = batch_results["depth_gt"].squeeze(1).numpy()
        depth_pred = batch_results["depth_pred"].squeeze(1).numpy()
        #depth_pred = depth_pred.view(images.shape[1], images.shape[2])

        for j in range(images.shape[0]):
            image_path = f"{result_dir}/{(i*BATCH_SIZE+j):03d}_image.png"
            detph_gt_path = f"{result_dir}/{(i*BATCH_SIZE+j):03d}_depth_gt.png"
            depth_pred_path = f"{result_dir}/{(i*BATCH_SIZE+j):03d}_depth_pred.png"

            save_img(images[j], image_path)
            save_img(depth_gt[j], detph_gt_path)
            save_img(depth_pred[j], depth_pred_path)

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

        image = Image.open(image_path).convert("RGB")
        depth = Image.open(depth_path).convert("L")

        if self.transform is not None:
            image = self.transform(image)
            depth = self.transform(depth)

        return image, depth



#############################
### Set up Neural Network ###
#############################
def depthestimation(output_dir, training, num_epoches):
    #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:4"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #config={'in_channels': 3, 'out_channels': 1, 'features': [64, 128, 256, 512]}
    model = oniroResnet.ResnetModel(num_in_layers=3)
    
    #input_data = torch.randn((BATCH_SIZE, 3, ))
    # Transformation-function for images 
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Set up loaders for training and test data
    train_dataset = DepthDataset(output_dir + "/train", transform=transform)
    valid_dataset = DepthDataset(output_dir + "/valid", transform=transform)
    test_dataset = DepthDataset(output_dir + "/test", transform=transform)

    # todo: adapt batch_size - 1
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print("type testdata : " + str(type(test_dataset)))

    # Define the loss function and optimizer 
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if args.training:
        train(model, device, train_loader, valid_loader, criterion, optimizer, num_epoches)
    results = test(model, device, test_loader, criterion)
    plot(results, output_dir)

if __name__ == "__main__":
    

    parser = argparse.ArgumentParser(description='neuronal depth estimator',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--helpme", "-help", action="help", help="Show the helper")
    parser.add_argument('--folder', '-f', help='folder, which contains test, train and depthfolders', required=True)
    parser.add_argument("--num_epoches", "-epoches", help="num of training-epoches", default=EPOCHES)
    parser.add_argument('--training', action=argparse.BooleanOptionalAction, help='Set/unset training', default=False)

    args = parser.parse_args()

    depthestimation(args.folder, True, args.num_epoches)


        