import argparse
import torch
import os

#from modules import models, residualmodels

#from timm import create_model
from torchsummary import summary
from torchvision.transforms import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

from modules.dataset import phosc_dataset
#from modules.engine import train_one_epoch, zslAccuracyTest
from modules.loss import PHOSCLoss

import torch.nn as nn

import os

os.getcwd()




#valid_csv="/media/aniketag/cd47fb8b-9ab3-460f-a88c-f0d67edf3ad8/home/k/phd/phdBackUp/phd/PHOSC-Zero-Shot-Word-Recognition-main/data/crop/iamSplit_Aspect_1024#10_05_2011#/val1.csv"
valid_csv="/global/D1/homes/aniket/data/IAM_Data1/iamSplit_Aspect_1024#10_05_2011#/val1.csv"
#valid_folder="/media/aniketag/cd47fb8b-9ab3-460f-a88c-f0d67edf3ad8/home/k/phd/phdBackUp/phd/PHOSC-Zero-Shot-Word-Recognition-main/data/crop/iamSplit_Aspect_1024#10_05_2011#/val"

valid_folder="/global/D1/homes/aniket/data/IAM_Data1/iamSplit_Aspect_1024#10_05_2011#/val/"

dataset_valid = phosc_dataset(valid_csv,None,
                              valid_folder, transforms.ToTensor())

data_loader_valid = torch.utils.data.DataLoader(
    dataset_valid,
    batch_size=16,
    num_workers=1,
    drop_last=False,
    shuffle=True
)



for samples, targets, words,path in data_loader_valid:

    #print(" path:",len(path))
    print(" samples:",samples.shape,"\t words:",words,"\t path:",path)

