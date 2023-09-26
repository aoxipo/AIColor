import os
import sys
import pickle
import cv2
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import pandas as pd
from skimage.transform import rotate

class ISICDataset(Dataset):
    def __init__(self, 
                 args, 
                 data_path , 
                 transform = None, 
                 mode = 'Training',
                 plane = False):


        df = pd.read_csv(os.path.join(data_path, 'ISBI2016_ISIC_Part3B_' + mode + '_GroundTruth.csv'), encoding='gbk')
        print("open:", os.path.join(data_path, 'ISBI2016_ISIC_Part3B_' + mode + '_GroundTruth.csv') )
        
        self.name_list = df.iloc[:,0].tolist()
        self.label_list = df.iloc[:,1].tolist()
        self.data_path = os.path.join(data_path, 'ISBI2016_ISIC_Part3B_'+ mode + "_Data/")
        self.mode = mode

        self.transform = transform

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, name) + ".jpg"
        
        mask_name = self.label_list[index]
        
        msk_path = os.path.join(self.data_path, name) + "_Segmentation.png"
        # print( self.data_path, img_path, msk_path, name, mask_name)
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')

        # if self.mode == 'Training':
        #     label = 0 if self.label_list[index] == 'benign' else 1
        # else:
        #     label = int(self.label_list[index])

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)
            mask = self.transform(mask)

        print("get data:", img.shape, mask.shape, name)
        return (img, mask, name)