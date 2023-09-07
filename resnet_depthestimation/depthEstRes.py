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
from loss import MonodepthLoss
import cv2

from torchvision.utils import save_image

BATCH_SIZE = 1
EPOCHES=10
B = 0.06499999761581421
F = 711.1112738715278283

def save_img(image, path):
    img = Image.fromarray((image  * 255).astype(np.uint8))
    img.save(path)

def train(model, device, train_loader, criterion, optimizer, num_epoches):

    model.to(device)

    for epoch in range(num_epoches):
        model.train()
        train_loss = 0.0
        for lefts, rights in train_loader:
            lefts = lefts.to(device)
            rights = rights.to(device)

            # reset gradients of all tensors to zero
            optimizer.zero_grad()

            outputs = model(lefts)
            # count = 0
            # for el in outputs:
            #     im = el.to("cpu").detach().squeeze(0).numpy()
            #     c1 = im[0,:,:]
            #     c2 = im[1,:,:]
                
            #     image1 = Image.fromarray((c1 * 255).astype(np.uint8))
            #     image2 = Image.fromarray((c2   * 255).astype(np.uint8))

            #     # Save the images
            #     image1.save("test" + str(count) +'_channel1.png')
            #     image2.save("test" + str(count) + "_channel2.png")
            #     count+=1
            # outputs_gray = torch.mean(outputs[0], dim=1, keepdim=True)
           
            loss = criterion(outputs, [lefts,rights])  
            # calculate gradients with respect to loss
            loss.backward()
            # update model parameters
            optimizer.step()

            train_loss += loss.item()
    
        # model.eval()
        # valid_loss = 0.0
        # for images, depths in valid_loader:
        #     images = images.to(device)
        #     depths = depths.to(device)

        #     outputs = model(images)
        #     outputs_gray = torch.mean(outputs[0], dim=1, keepdim=True)
        #     loss = criterion(outputs_gray, depths)
        #     valid_loss += loss.item()

        # print(f"Epoch {epoch + 1}/{num_epoches} - Loss: {train_loss / len(train_loader):.4f} \t\t Validation Loss: {valid_loss / len(valid_loader)}")

    torch.save(model.state_dict(), "model.pth")

def post_process_disparity(disp):
    (_, h, w) = disp.shape
    l_disp = disp[0, :, :]
    r_disp = np.fliplr(disp[1, :, :])
    m_disp = 0.5 * (l_disp + r_disp)
    (l, _) = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

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
            print("OOO")
            print(outputs[0])
            # todo postprocessing

            #outputs_gray = torch.mean(outputs[0], dim=1, keepdim=True)

            mask = torch.zeros(1,2,512,512).to(device)
            mask[0,0,:,:] = B*F
            mask[:,:,0,0] = 1.0
            depth_from_disparity =  (mask/ (outputs[0]*255))/255
            # print("dfd")
            # print(depth_from_disparity)
            # print(np.max(depth_from_disparity.cpu().numpy()))
            # print("outputs")
            # print(outputs[0])
            # print(np.max(outputs[0].cpu().numpy()))
            
            # store image, estimated depth and true depth for plotting
            batch_results = {
                "images": images.cpu(),
                "depth_gt": depths.cpu(),
                "depth_pred": depth_from_disparity.cpu(),
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
        depth_gt = batch_results["depth_gt"].squeeze(1).permute(0,2,3,1).numpy()
        depth_pred = batch_results["depth_pred"].permute(0,2,3,1).numpy()
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

class DepthDatasetStereo(Dataset):
    def __init__(self, data_path_a, data_path_b, transform=None):
        self.data_path_a = data_path_a
        self.data_path_b = data_path_b
        self.transform = transform
        self.image_filenames = self.find_common_images()

    def find_common_images(self):
        image_filenames_a = set(os.listdir(self.data_path_a))
        image_filenames_b = set(os.listdir(self.data_path_b))
        common_images = list(image_filenames_a.intersection(image_filenames_b))
        return common_images

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        img_name = self.image_filenames[index]
        img_path_a = os.path.join(self.data_path_a, img_name)
        img_path_b = os.path.join(self.data_path_b, img_name)

        img_a = Image.open(img_path_a).convert("RGB")
        img_b = Image.open(img_path_b).convert("RGB")

        if self.transform is not None:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)
        
        return img_a, img_b



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
    train_dataset = DepthDatasetStereo(output_dir + "/train", output_dir + "/disparity", transform=transform)
    #valid_dataset = DepthDataset(output_dir + "/valid", transform=transform)
    test_dataset = DepthDatasetStereo(output_dir + "/test", output_dir + "/disparity_depth", transform=transform)

    # todo: adapt batch_size - 1
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    #valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print("type testdata : " + str(type(test_dataset)))

    # Define the loss function and optimizer 
    #criterion = nn.MSELoss()
    #optimizer = optim.Adam(model.parameters(), lr=0.001)

    # https://github.com/OniroAI/MonoDepth-PyTorch/blob/master/main_monodepth_pytorch.py
    criterion = MonodepthLoss(
                n=4,
                SSIM_w=0.85,
                disp_gradient_w=0.1, lr_w=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    if args.training:
        train(model, device, train_loader, criterion, optimizer, num_epoches)
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


        