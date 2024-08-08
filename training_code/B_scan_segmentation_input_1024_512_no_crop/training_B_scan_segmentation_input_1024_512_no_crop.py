from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform, util, color
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms, utils
import matplotlib.pyplot as plt
import time
import copy
import torch.nn.functional as F
import tqdm
from mpl_toolkits.mplot3d import Axes3D
import albumentations as abm
import cv2
import torchvision.transforms.functional as TF
import scipy
from sklearn.model_selection import train_test_split
import pathlib
from pathlib import Path
import warnings; warnings.simplefilter('ignore')

class Dataset(Dataset):
    """B_scan_segmentation dataset Preparation."""

    def __init__(self, inputs, targets, transform=None):
        """
        Args:
            root_dir (string): Directory containing image folders.
            keyword: examples are "img_train"
        """
        self.inputs = inputs
        self.targets = targets
        # print("self.inputs", self.inputs)
        # print("self.targets", self.targets)
        self.transform = transform

        
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_ID = self.inputs[idx]
        target_ID = self.targets[idx]
        image = io.imread(input_ID)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        mask = io.imread(target_ID)
        mask = mask[:, :, 0]
        mask[mask==3] = 0
        mask[mask==4] = 0
        
        sample = {'image': image, 'mask':mask}
        
        if self.transform:
            sample = self.transform(sample)

        return sample
    
# Data augmentation - Must be applied in the order as follows
class ShiftScaleRotate(object):
    def __init__(self, p = 0.8):
        self.p = p
        self.ssr = abm.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=8,
                                    interpolation=1, border_mode=4,
                                    value=None, mask_value=None,
                                    p=self.p) # always_apply=True

    def __call__(self, sample):
        image = sample['image']
        mask = sample['mask']
        transformed = self.ssr(image = image, mask = mask)
        image = transformed['image']
        mask = transformed['mask']

        return{'image':image,
               'mask':mask
        }

class Blur(object):  
    def __init__(self, p = 0.8):
        self.p = p
        self.composed_functions = abm.Blur(p=self.p)

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
    
        transformed = self.composed_functions(image = image, mask = mask)
        image = transformed['image']
        label = transformed['mask']
        
        return{'image':image,
               'mask':mask,
        }

class Jitter(object):
    def __init__(self, p = 0.2):
        self.p = p
        self.jitter = torchvision.transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05)
    
    def __call__(self, sample):
        
        image, mask = sample['image'], sample['mask']
        
        # jitter 20 percent of the time                                            
        if (np.random.uniform() < self.p):
            image = TF.to_pil_image(image)
            image = self.jitter(image)

            image = np.array(image)
        
        return{'image' : image,
               'mask' : mask
        }
    
class AddNoiseAddDropout(object):  
    def __init__(self, p = 0.8):
        self.p = p
        self.dropout_function = abm.CoarseDropout(max_holes = 140, max_height = 5, 
                                                  max_width = 5, always_apply = True)
        self.noise_function = abm.GaussNoise(always_apply=True)
        self.composed_functions = abm.Compose([self.dropout_function, self.noise_function])
        
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        
        # add noise and dropout 20 percent of the time                 
        if (np.random.uniform() < self.p):
            image = {'image': image}
            image = self.composed_functions(**image)
            image = image['image']
        
        
        return{'image' : image,
               'mask' : mask
        }
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, mask = [sample['image'], sample['mask']]
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = TF.to_tensor(image)
        mask = torch.from_numpy(mask).type(torch.long)
        
        return{'image':image, 'mask': mask}

def get_filenames_of_input(path, total):
    """Returns a list of files in a directory/path. Uses pathlib."""
    filenames = []
    for i in range(total):
        cur = path / Path("frame{:06d}".format(i) + ".jpg")
        filenames.append(cur)
    return filenames

def get_filenames_of_mask(path, total):
    """Returns a list of files in a directory/path. Uses pathlib."""
    filenames = []
    for i in range(total):
        cur = path / Path("frame{:06d}".format(i) + ".png")
        filenames.append(cur)
    return filenames

def save_checkpoint(state, filename):
    torch.save(state, filename)

# Create resnet
class OCT(torch.nn.Module):
  """
  output size is 3 x 1024 x 512
  """
  def __init__(self, orig_resnet):
    super().__init__()
    self.orig_resnet = orig_resnet
    
    self.dec4 = nn.ConvTranspose2d(512, 64, 2, 2)
    self.conv4_1 = nn.Conv2d(320, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
    self.batchnorm_4_1 = torch.nn.BatchNorm2d(64)
    self.conv4_2 = nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
    self.batchnorm_4_2 = torch.nn.BatchNorm2d(64)
    
    
    # decoder 
    self.dec1 = nn.ConvTranspose2d(64, 64, kernel_size=(4, 4), stride=(4,4))
    # dimension preserving convolution
    self.conv1_1 = nn.Conv2d(128, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
    self.batchnorm_1_1 = torch.nn.BatchNorm2d(64)
    self.conv1_2 = nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
    self.batchnorm_1_2 = torch.nn.BatchNorm2d(64)
    
    # another decoder layer
    self.dec2 = nn.ConvTranspose2d(64, 32, kernel_size = (2,2), stride=(2,2))
    self.conv2_1 = nn.Conv2d(96, 32, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
    self.batchnorm_2_1 = torch.nn.BatchNorm2d(32)
    self.conv2_2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
    self.batchnorm_2_2 = torch.nn.BatchNorm2d(32) 
    
    # second decoder layer
    self.dec3 = nn.ConvTranspose2d(32, 16, 2, 2)
    self.conv3_1 = nn.Conv2d(16, 16, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
    self.batchnorm_3_1 = torch.nn.BatchNorm2d(16)
    self.conv3_2 = nn.Conv2d(16, 3, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
    self.batchnorm_3_2 = torch.nn.BatchNorm2d(3)
    self.conv3_3 = nn.Conv2d(3, 3, kernel_size=(1,1), stride=(1,1), padding=(0,0), bias=False)
    
  def forward(self, x):
    # encoder
    x1 = self.orig_resnet.conv1(x) # 64 x 512 x 256
    x2 = self.orig_resnet.bn1(x1) 
    x3 = self.orig_resnet.relu(x2)
    x4 = self.orig_resnet.maxpool(x3) # 64 x 256 x 128
    x5 = self.orig_resnet.layer1(x4) # 64 x 256 x 128
    x6 = self.orig_resnet.layer2(x5) # 128 x 128 x 64
    x7 = self.orig_resnet.layer3(x6) # 256 x 64 x 32
    x8 = self.orig_resnet.layer4(x7) # 512 x 32 x 16
    
    x13 = self.dec4(x8) # 64 x 64 x 32
    x13 = torch.cat((x13, x7), dim=1) # 320
    x13 = self.batchnorm_4_1( self.conv4_1(x13) ) # 64
    x13 = F.relu(x13)
    x13 = self.batchnorm_4_2(self.conv4_2(x13))
    x13 = F.relu(x13) # 64
    
    # decoder
    x13 = self.dec1(x13) # 64 x 256 x 128
    x13 = torch.cat((x13, x5), dim=1) # 128
    x13 = self.batchnorm_1_1( self.conv1_1(x13) ) # 64
    x13 = F.relu(x13)
    x13 = self.batchnorm_1_2(self.conv1_2(x13))
    x13 = F.relu(x13) # 64

    x14 = self.dec2(x13) # 32 x 512 x 256
    x14 = torch.cat((x14, x3), dim=1) # 96
    x14 = self.batchnorm_2_1( self.conv2_1(x14) ) # 32
    x14 = F.relu(x14)
    x14 = self.batchnorm_2_2(self.conv2_2(x14))
    x14 = F.relu(x14)
    
    x15 = self.dec3(x14) # 16 x 1024 x 512
    x15 = self.batchnorm_3_1(self.conv3_1(x15)) # 16
    x15 = F.relu(x15)
    x15 = self.batchnorm_3_2( self.conv3_2(x15) )
    x15 = F.relu(x15)
    x15 = self.conv3_3(x15)
    return x15
  
def train_model(model, criterion, optimizer, scheduler=None, num_epochs=25, device=None):
    since = time.time()
    
    # initialization every epoch
    best_validation_model_output = None
    best_model_wts = copy.deepcopy(model.state_dict())

    best_loss = 1000000

    best_acc = -1000

    train_losses = [] 
    test_losses = []
    final_accuracies = []
    
    running_loss_avg = None
    all_info = []
    print("im here")
    # no need to resample validation loader every epoch
    validation_dataloader = DataLoader(validation_dataset, batch_size = 32, shuffle = False)
    for epoch in range(num_epochs):
        # sample from dataloader every epoch
        train_dataloader = DataLoader(train_dataset, batch_size = 32, shuffle=True)
        dataloaders = {'train': train_dataloader, 'val': validation_dataloader}
        
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        error_per_sample_for_epoch = 0.0
        for phase in ['train', 'val']:
            count = 0
            running_loss = 0.0
            running_loss_avg = 0.0
            absolute_error_list = []
            
            model_outputs = []
            
            if phase == 'train':
                if scheduler is not None:
                    scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode   

            # Iterate over data.
            progress_bar = tqdm.tqdm(dataloaders[phase], desc=phase)
            for batch_idx, data in enumerate(progress_bar):
                inputs = data['image'].to(device).float()
                masks = data['mask'].to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, masks)

                    # calculate error
                    outputs = outputs.cpu().detach().numpy()
                    count += outputs.shape[0]
                    outputs = np.argmax(outputs, axis = 1)
                    masks = masks.cpu().detach().numpy()
                    error = outputs - masks
                    absolute_error = np.sum(np.absolute(error))
                        
                    absolute_error_list.append(absolute_error)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Loss
                running_loss += loss.item() # accumulate loss across batches
                running_loss_avg = running_loss / (batch_idx + 1)
                # error per sample upto s ofar
                running_error = np.sum(absolute_error_list) / count


                progress_bar.set_description(
                'loss avg by batch: {:.4}, running_abs_error: {:.2f}'.format(
                    running_loss / (batch_idx + 1), running_error) )
            
            # record train/test loss
            if phase == 'train':
                train_losses.append(running_loss)
                all_info.append([epoch, "train", running_loss_avg, running_error])
                
            if phase == 'val':
                test_losses.append(running_loss)
                all_info.append([epoch, "val", running_loss_avg, running_error])

        # per epoch loop
        # save model_parameters every epoch
        if not os.path.exists("./segmentation_parameters"):
            # Create the directory
            os.makedirs("./segmentation_parameters")
        save_checkpoint({'state_dict': model.state_dict(), 'optimizer':optimizer.state_dict()},
                       "./segmentation_parameters/epoch_" + str(epoch) + phase + "_checkpoint.pth.tar")
        
        np.savetxt("progress_log.csv", np.asarray(all_info), delimiter=",", fmt="%s")
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best acc: {:4f}, Best loss: {:4f}'.format(best_acc, best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_validation_model_output, train_losses, test_losses

if __name__=="__main__": 
    # Create Train/Validation Dataset
    root = pathlib.Path.cwd() / '../../dataset/B_scan_segmentation_input_1024_512_no_crop'
    total_img = 5 # modified based on your data size

    # input and target files
    inputs = get_filenames_of_input(root / 'input', total_img)
    masks = get_filenames_of_mask(root / 'mask', total_img)

    # random seed
    random_seed = 3

    # split dataset into training set and validation set
    train_size = int(0.8 * total_img)  # 4:1 split

    inputs_train, inputs_valid, targets_train, targets_valid = train_test_split(
        inputs, masks,
        random_state=random_seed,
        train_size=train_size,
        shuffle=True)

    # targets_train, targets_valid = train_test_split(
    #     masks,
    #     random_state=random_seed,
    #     train_size=train_size,
    #     shuffle=True)

    train_datasets = []
    validation_datasets = []

    # train dataset
    data_transform_train = transforms.Compose([
                ShiftScaleRotate(),
    #             Blur(),
                Jitter(),
                AddNoiseAddDropout(),
                ToTensor()
                ])

    data_transform_test = transforms.Compose([
                ToTensor()
                ])
    train_dataset = Dataset(inputs=inputs_train, targets=targets_train, 
                            transform = data_transform_train) #= data_transform)

    # validation dataset
    validation_dataset = Dataset(inputs=inputs_valid, targets=targets_valid,  
                        transform = data_transform_test) #= data_transform)
    print("train_dataset tot images:", len(train_dataset))
    print("validation_dataset tot images:", len(validation_dataset))
    print("train percentage: ", len(train_dataset)/(len(train_dataset) + len(validation_dataset)))
    print("validation percentage: ", len(validation_dataset)/(len(train_dataset) + len(validation_dataset)))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    resnet = models.resnet18(pretrained=False)
    n_input_channel = 1
    resnet.conv1 = torch.nn.Conv2d(n_input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # choose resnet of your choice
    resnet_enc_dec_concat_more = OCT(resnet) 
    model_ft = resnet_enc_dec_concat_more.to(device)
    # choose criteria
    weights = [0.002, 0.999, 0.999]
    class_weights = torch.FloatTensor(weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer_ft = optim.RMSprop(model_ft.parameters(), lr=0.00001, weight_decay=1e-8, momentum=0.9)

    # OCT segmentation 840 training images
    model_ft, best_validation_model_output, train_losses, test_losses = train_model(
                model = model_ft,
                criterion = criterion,
                optimizer = optimizer_ft,
                num_epochs = 200, 
                device=device)