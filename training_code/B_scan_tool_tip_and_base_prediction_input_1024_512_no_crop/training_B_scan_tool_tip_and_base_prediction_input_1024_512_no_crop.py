from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform, util
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
import random

class Dataset(Dataset):
    """B_scan_tool_tip_and_base_prediction dataset Preparation."""

    def __init__(self, img_dir, csv_dir, transform=None):
        """
        Args:
            root_dir (string): Directory containing image folders.
            keyword: examples are "img_train"
        """
        self.img_dir = img_dir
        self.csv_dir = csv_dir
        self.labels = pd.read_csv(self.csv_dir, header = None)
        print("self.labels", self.labels)
        self.transform = transform

        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        # get label
        label = np.reshape(self.labels.iloc[idx,:].values, (2, 2))
        label = label.astype('float')
        v = label[1] - label[0]
        unit_v = v / np.linalg.norm(v)
        distance = 100
        label[1][0] = int(label[0][0] + unit_v[0] * distance)
        label[1][1] = int(label[0][1] + unit_v[1] * distance)
        
        # get relevant frame
        img_path = os.path.join(self.img_dir, "frame{:06d}".format(idx) + ".jpg")
        image = io.imread(img_path)
        sample = {'image': image, 'label':label}
        
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
        self.composed_functions = abm.Compose([self.ssr], keypoint_params=abm.KeypointParams(format='xy', remove_invisible=True))

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        transformed = self.composed_functions(image = image, keypoints = label)
        image = transformed['image']
        label = transformed['keypoints']

        return{'image':image,
               'label':label
        }
    
class SafeRotate(object):  
    def __init__(self, p = 1):
        self.p = p
        self.ssr = abm.SafeRotate(limit=[90, 90], interpolation=1, 
                                   border_mode=4, value=None, 
                                   mask_value=None, always_apply=True, p=self.p)
        self.composed_functions = abm.Compose([self.ssr], keypoint_params=abm.KeypointParams(format='xy'))      
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        transformed = self.composed_functions(image = image, keypoints = label)
        image = transformed['image']
        label = transformed['keypoints']

        return{'image':image,
               'label':label,
        }

class Crop(object):
    def __init__(self, x_min=0, y_min=0, x_max=700, y_max=500, always_apply = False, p = 1.0):
        self.p = p
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.CenterCrop = abm.Crop(x_min= self.x_min, y_min= self.y_min, x_max= self.x_max, y_max= self.y_max, p = self.p)
        self.composed_functions = abm.Compose([self.CenterCrop], keypoint_params=abm.KeypointParams(format='xy'))
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        transformed = self.composed_functions(image = image, keypoints = label)
        image = transformed['image']
        label = transformed['keypoints']
        
        return{'image':image,
               'label':label
        }
    
class CenterCrop(object):
    def __init__(self, height = 480, width = 640, always_apply = False, p = 1.0):
        self.p = p
        self.height = height
        self.width = width
        self.CenterCrop = abm.CenterCrop(height = self.height, width = self.width, p = self.p)
        self.composed_functions = abm.Compose([self.CenterCrop], keypoint_params=abm.KeypointParams(format='xy'))
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        transformed = self.composed_functions(image = image, keypoints = label)
        image = transformed['image']
        label = transformed['keypoints']
        
        return{'image':image,
               'label':label
        }

class Blur(object):  
    def __init__(self, p = 0.8):
        self.p = p
        self.composed_functions = abm.Compose(
            [abm.Blur(p=self.p)], 
            keypoint_params=abm.KeypointParams(format='xy')
        )

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
    
        transformed = self.composed_functions(image = image, keypoints = label)
        image = transformed['image']
        label = transformed['keypoints']
        
        return{'image':image,
               'label':label,
        }

class Jitter(object):
    def __init__(self, p = 0.8):
        self.p = p
        self.jitter = torchvision.transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25)
    
    def __call__(self, sample):
        
        image, label = sample['image'], sample['label']
        
        # jitter 20 percent of the time                                            
        if (np.random.uniform() < self.p):
            image = TF.to_pil_image(image)
            image = self.jitter(image)

            image = np.array(image)
        
        
        return{'image':image,
               'label':label
        }

    
class AddHueSaturationRGBShift(object):  
    def __init__(self, p = 0.8):
        self.p = p
        self.hue_saturation = abm.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20)
        self.rgb_shift = abm.RGBShift (r_shift_limit=20, g_shift_limit=20, b_shift_limit=20)
        self.composed_functions = abm.Compose([self.hue_saturation, self.rgb_shift])
        
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        # add noise and dropout 20 percent of the time                 
        if (np.random.uniform() < self.p):
            image = {'image': image}
            image = self.composed_functions(**image)
            image = image['image']
        
        
        return{'image':image,
               'label':label
        }
    
class AddNoiseAddDropout(object):  
    def __init__(self, p = 0.8):
        self.p = p
        self.dropout_function = abm.CoarseDropout(max_holes = 140, max_height = 5, 
                                                  max_width = 5, always_apply = True)
        self.noise_function = abm.GaussNoise(always_apply=True)
        self.composed_functions = abm.Compose([self.dropout_function, self.noise_function])
        
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        # add noise and dropout 20 percent of the time                 
        if (np.random.uniform() < self.p):
            image = {'image': image}
            image = self.composed_functions(**image)
            image = image['image']
        
        
        return{'image':image,
               'label':label
        }
    
class ConvertToolTipPositionToImageLabel(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = [sample['image'], sample['label']]
        # create label
        label = np.array(label)
        label = label.astype('float')
        v = label[1] - label[0]
        unit_v = v / np.linalg.norm(v)
        distance = 100
        label[1][0] = int(label[0][0] + unit_v[0] * distance)
        label[1][1] = int(label[0][1] + unit_v[1] * distance)
        if (label[0][0] < 0):
            label[0][0] = 0
        if (label[0][0] > 512):
            label[0][0] = 512
        if (label[0][1] < 0):
            label[0][1] = 0
        if (label[0][1] > 1024):
            label[0][1] = 1024
        if (label[1][0] < 0):
            label[1][0] = 0
        if (label[1][0] > 512):
            label[1][0] = 512
        if (label[1][1] < 0):
            label[1][1] = 0
        if (label[1][1] > 1024):
            label[1][1] = 1024
        zeros1 = np.zeros((1024, 512))
        coord_x1 = int(label[0][0])
        coord_y1 = int(label[0][1])
        zeros1[coord_y1, coord_x1] = 1
        class_label1 = np.argmax(zeros1)
        zeros2 = np.zeros((1024, 512))
        coord_x2 = int(label[1][0])
        coord_y2 = int(label[1][1])
        zeros2[coord_y2, coord_x2] = 1
        class_label2 = np.argmax(zeros2)
        class_label = []
        class_label.append(class_label1)
        class_label.append(class_label2)
        # rounded coordinates to resized image
        label = np.asarray((coord_x1, coord_y1, coord_x2, coord_y2))
        
        return{'image':image, 'label': label, 'class_label': class_label} 
    
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label, class_label = [sample['image'], sample['label'], sample['class_label']]
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = TF.to_tensor(image)
        class_label = torch.tensor(class_label)
        
        return{'image':image, 'label': label, 'class_label': class_label} 

def save_checkpoint(state, filename):
    torch.save(state, filename)

# Create resnet
class Resnet_tool_tip_labelling_large(torch.nn.Module):
  """
  output size is 1024 x 512
  """
  def __init__(self, orig_resnet):
    super().__init__()
    self.orig_resnet = orig_resnet
    
    self.dec4 = nn.ConvTranspose2d(512, 64, 2, 2)
    self.conv4_1 = nn.Conv2d(320, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
    self.batchnorm_4_1 = torch.nn.BatchNorm2d(64)
    self.conv4_2 = nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
    self.batchnorm_4_2 = torch.nn.BatchNorm2d(64)
    
    
    # decoder 1
    self.dec11 = nn.ConvTranspose2d(64, 64, kernel_size=(4, 4), stride=(4,4))
    # dimension preserving convolution
    self.conv11_1 = nn.Conv2d(128, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
    self.batchnorm1_1_1 = torch.nn.BatchNorm2d(64)
    self.conv11_2 = nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
    self.batchnorm1_1_2 = torch.nn.BatchNorm2d(64)
    
    # another decoder layer
    self.dec12 = nn.ConvTranspose2d(64, 32, kernel_size = (2,2), stride=(2,2))
    self.conv12_1 = nn.Conv2d(96, 32, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
    self.batchnorm1_2_1 = torch.nn.BatchNorm2d(32)
    self.conv12_2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
    self.batchnorm1_2_2 = torch.nn.BatchNorm2d(32) 
    
    # second decoder layer
    self.dec13 = nn.ConvTranspose2d(32, 16, 2, 2)
    self.conv13_1 = nn.Conv2d(16, 16, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
    self.batchnorm1_3_1 = torch.nn.BatchNorm2d(16)
    self.conv13_2 = nn.Conv2d(16, 1, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
    self.batchnorm1_3_2 = torch.nn.BatchNorm2d(1)
    self.conv13_3 = nn.Conv2d(1, 1, kernel_size=(1,1), stride=(1,1), padding=(0,0), bias=False)
    
    # decoder 2
    self.dec21 = nn.ConvTranspose2d(64, 64, kernel_size=(4, 4), stride=(4,4))
    # dimension preserving convolution
    self.conv21_1 = nn.Conv2d(128, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
    self.batchnorm2_1_1 = torch.nn.BatchNorm2d(64)
    self.conv21_2 = nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
    self.batchnorm2_1_2 = torch.nn.BatchNorm2d(64)
    
    # another decoder layer
    self.dec22 = nn.ConvTranspose2d(64, 32, kernel_size = (2,2), stride=(2,2))
    self.conv22_1 = nn.Conv2d(96, 32, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
    self.batchnorm2_2_1 = torch.nn.BatchNorm2d(32)
    self.conv22_2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
    self.batchnorm2_2_2 = torch.nn.BatchNorm2d(32) 
    
    # second decoder layer
    self.dec23 = nn.ConvTranspose2d(32, 16, 2, 2)
    self.conv23_1 = nn.Conv2d(16, 16, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
    self.batchnorm2_3_1 = torch.nn.BatchNorm2d(16)
    self.conv23_2 = nn.Conv2d(16, 1, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
    self.batchnorm2_3_2 = torch.nn.BatchNorm2d(1)
    self.conv23_3 = nn.Conv2d(1, 1, kernel_size=(1,1), stride=(1,1), padding=(0,0), bias=False)
    
    
  def forward(self, x):
    # encoder
    x1 = self.orig_resnet.conv1(x) # 64 x 512 x 256
    # print("x1", x1.shape)
    x2 = self.orig_resnet.bn1(x1) 
    # print("x2", x2.shape)
    x3 = self.orig_resnet.relu(x2)
    # print("x3", x3.shape)
    x4 = self.orig_resnet.maxpool(x3) # 64 x 256 x 128
    # print("x4", x4.shape)
    x5 = self.orig_resnet.layer1(x4) # 64 x 256 x 128
    # print("x5", x5.shape)
    x6 = self.orig_resnet.layer2(x5) # 128 x 128 x 64
    # print("x6", x6.shape)
    x7 = self.orig_resnet.layer3(x6) # 256 x 64 x 32
    # print("x7", x7.shape)
    x8 = self.orig_resnet.layer4(x7) # 512 x 32 x 16
    
    x13 = self.dec4(x8) # 64 x 64 x 32
    x13 = torch.cat((x13, x7), dim=1) # 320
    x13 = self.batchnorm_4_1( self.conv4_1(x13) ) # 64
    x13 = F.relu(x13)
    x13 = self.batchnorm_4_2(self.conv4_2(x13))
    x13 = F.relu(x13) # 64
    
    # decoder 1
    x113 = self.dec11(x13) # 64 x 256 x 128
    x113 = torch.cat((x113, x5), dim=1) # 128
    x113 = self.batchnorm1_1_1( self.conv11_1(x113) ) # 64
    x113 = F.relu(x113)
    x113 = self.batchnorm1_1_2(self.conv11_2(x113))
    x113 = F.relu(x113) # 64

    x114 = self.dec12(x113) # 32 x 512 x 256
    x114 = torch.cat((x114, x3), dim=1) # 96
    x114 = self.batchnorm1_2_1( self.conv12_1(x114) ) # 32
    x114 = F.relu(x114)
    x114 = self.batchnorm1_2_2(self.conv12_2(x114))
    x114 = F.relu(x114)
    
    x115 = self.dec13(x114) # 16 x 1024 x 512
    x115 = self.batchnorm1_3_1(self.conv13_1(x115)) # 16
    x115 = F.relu(x115)
    x115 = self.batchnorm1_3_2( self.conv13_2(x115) )
    x115 = F.relu(x115)
    x115 = self.conv13_3(x115)
    
    # decoder 2
    x213 = self.dec21(x13) # 64 x 256 x 128
    x213 = torch.cat((x213, x5), dim=1) # 128
    x213 = self.batchnorm2_1_1( self.conv21_1(x213) ) # 64
    x213 = F.relu(x213)
    x213 = self.batchnorm2_1_2(self.conv21_2(x213))
    x213 = F.relu(x213) # 64

    x214 = self.dec22(x213) # 32 x 512 x 256
    x214 = torch.cat((x214, x3), dim=1) # 80
    x214 = self.batchnorm2_2_1( self.conv22_1(x214) ) # 32
    x214 = F.relu(x214)
    x214 = self.batchnorm2_2_2(self.conv22_2(x214))
    x214 = F.relu(x214)
    
    x215 = self.dec23(x214) # 16 x 1024 x 512
    x215 = self.batchnorm2_3_1(self.conv23_1(x215)) # 16
    x215 = F.relu(x215)
    x215 = self.batchnorm2_3_2( self.conv23_2(x215) )
    x215 = F.relu(x215)
    x215 = self.conv23_3(x215) # 1 x 1024 x 512
    
    return x115, x215
 
def train_model(model, criterion, criterion2, optimizer, scheduler=None, num_epochs=25, device=None):
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
    validation_dataloader = DataLoader(validation_dataset, batch_size = 24, shuffle = False)
    for epoch in range(num_epochs):
        
        # sample from dataloader every epoch
        train_dataloader = DataLoader(train_dataset, batch_size = 24, shuffle=True)
        dataloaders = {'train': train_dataloader, 'val': validation_dataloader}
        
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        error_per_sample_for_epoch = 0.0
        for phase in ['train', 'val']:
    
            running_loss = 0.0
            running_loss_avg = 0.0
            absolute_error_list = []
            absolute_error_list_x = []
            absolute_error_list_y = []
            absolute_error_list_z = []
            
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
                labels = data['label'].to(device)
                argmax_labels = data['class_label'].to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs1, outputs2 = model(inputs)
                    outputs_flatten1 = outputs1.view(outputs1.size(0), -1)
                    outputs_flatten2 = outputs2.view(outputs2.size(0), -1)
                    current_argmax1 = np.array(np.unravel_index(outputs_flatten1.cpu().argmax(axis=1), (1024,512)))
                    current_argmax2 = np.array(np.unravel_index(outputs_flatten2.cpu().argmax(axis=1), (1024,512)))
                    l3label = np.ones((outputs1.size(0),)) * 100
                    l3label = torch.from_numpy(l3label)
                    l3output = np.linalg.norm(current_argmax1 - current_argmax2, axis = 0)
                    l3output = torch.from_numpy(l3output)
                    loss1 = criterion(outputs_flatten1, argmax_labels[:,0])
                    loss2 = criterion(outputs_flatten2, argmax_labels[:,1])
                    loss3 = criterion2(l3output, l3label)
                    loss = loss1 + loss2 + loss3
                    # calculate error
                    outputs_np1 = outputs1.cpu().detach().numpy()
                    outputs_np2 = outputs2.cpu().detach().numpy()
                    answer1 = labels[:,0:2].cpu().numpy() # this is xy position (not class_label, which is one number)
                    answer2 = labels[:,2:4].cpu().numpy()
                    # print("answer shape", answer.shape)
                    for i in range(len(outputs_np1)):
                        # get current output
                        a = outputs_np1[i,:,:,:]
                        # find the argmax and reshape into xy coords
                        current_argmax = np.unravel_index(a.argmax(), (1024,512))
                        # get the correct xy coords
                        current_answer = answer1[i,:]
                        # switch the positions of the xy coords, since it was found to be switched from prior tests
                        current_argmax_switch = np.array((current_argmax[1], current_argmax[0]))
                        error = current_answer - current_argmax_switch
#                         error = current_answer - current_argmax
                        absolute_error = np.sum(np.absolute(error))
                        
                        absolute_error_list.append(absolute_error)
                        absolute_error_list_x.append(np.absolute(error)[0])
                        absolute_error_list_y.append(np.absolute(error)[1])

                    for i in range(len(outputs_np2)):
                        # get current output
                        a = outputs_np2[i,:,:,:]
                        # find the argmax and reshape into xy coords
                        current_argmax = np.unravel_index(a.argmax(), (1024,512))
                        # get the correct xy coords
                        current_answer = answer2[i,:]
                        # switch the positions of the xy coords, since it was found to be switched from prior tests
                        current_argmax_switch = np.array((current_argmax[1], current_argmax[0]))
                        error = current_answer - current_argmax_switch
                        absolute_error = np.sum(np.absolute(error))
                        
                        absolute_error_list.append(absolute_error)
                        absolute_error_list_x.append(np.absolute(error)[0])
                        absolute_error_list_y.append(np.absolute(error)[1])
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Loss
                running_loss += loss.item() # accumulate loss across batches
                running_loss_avg = running_loss / (batch_idx + 1)
                # error per sample upto s ofar
                running_error = np.sum(absolute_error_list) / len(absolute_error_list)
                running_error_x = np.sum(absolute_error_list_x) / len(absolute_error_list_x)
                running_error_y = np.sum(absolute_error_list_y) / len(absolute_error_list_y)

                progress_bar.set_description(
                'loss avg by batch: {:.4}, running_abs_error: {:.2f}, err_x: {:.2f}, err_y: {:.2f}'.format(
                    running_loss / (batch_idx + 1), running_error, running_error_x, running_error_y) )
                
            # record train/test loss
            if phase == 'train':
                train_losses.append(running_loss)
                all_info.append([epoch, "train", running_loss_avg, running_error_x, running_error_y])
                
            if phase == 'val':
                test_losses.append(running_loss)
                all_info.append([epoch, "val", running_loss_avg, running_error_x, running_error_y])

        # per epoch loop
        # save model_parameters every epoch
        if not os.path.exists("./tooltip_parameters"):
            # Create the directory
            os.makedirs("./tooltip_parameters")
        save_checkpoint({'state_dict': model.state_dict(), 'optimizer':optimizer.state_dict()},
                       "./tooltip_parameters/epoch_" + str(epoch) + phase + "_checkpoint.pth.tar")
        
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
    datasets = []
    train_datasets = []
    validation_datasets = []
    data_transform_train = transforms.Compose([
                ShiftScaleRotate(),
                Blur(),
                Jitter(),
                AddNoiseAddDropout(),
                ConvertToolTipPositionToImageLabel(),
                ToTensor()
                ])

    data_transform_validation = transforms.Compose([
                ConvertToolTipPositionToImageLabel(),
                ToTensor()
                ])
    train_dataset = Dataset(img_dir = "../../dataset/B_scan_tool_tip_and_base_prediction_input_1024_512_no_crop/train_img", csv_dir = "../../dataset/B_scan_tool_tip_and_base_prediction_input_1024_512_no_crop/train_tool_tip_and_base_points.csv",  
                            transform = data_transform_train) #= data_transform)

    validation_dataset = Dataset(img_dir = "../../dataset/B_scan_tool_tip_and_base_prediction_input_1024_512_no_crop/validation_img", csv_dir = "../../dataset/B_scan_tool_tip_and_base_prediction_input_1024_512_no_crop/validation_tool_tip_and_base_points.csv",  
                        transform = data_transform_validation) #= data_transform)
    print("train_dataset tot images:", len(train_dataset))
    print("validation_dataset tot images:", len(validation_dataset))
    print("train percentage: ", len(train_dataset)/(len(train_dataset) + len(validation_dataset)))
    print("validation percentage: ", len(validation_dataset)/(len(train_dataset) + len(validation_dataset)))
    # setup model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    resnet = models.resnet18(pretrained=False)
    # choose resnet of your choice
    resnet_enc_dec_concat_more = Resnet_tool_tip_labelling_large(resnet)
    model_ft = resnet_enc_dec_concat_more.to(device)
    # choose criteria
    criterion = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0003)
    # Total image: 1050
    model_ft, best_validation_model_output, train_losses, test_losses = train_model(
                model = model_ft,
                criterion = criterion,
                criterion2 = criterion2,
                optimizer = optimizer_ft,
                num_epochs = 112, 
                device=device)