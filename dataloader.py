import random
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
# from sklearn.model_selection import train_test_split
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import glob


class Dataload(Dataset):
    def __init__(self, file_path, gt_path, image_shape=(224, 224), data_type='train'):
        self.total_number = None

        self.file_path = file_path
        self.gt_path = gt_path

        self.image_shape = image_shape
        self.data_type = data_type

        self.photo_set = []
        self.set_gan()
        if data_type == 'train':
            self.load_data(file_path, gt_path)
        else:
            self.load_data_new(file_path)
        
    def check_dir(self, path):
        return os.path.exists(path)

    def read_image_data(self, file_path):
        image = cv2.imread(file_path)
        if(image is None):
            return None
            #raise RuntimeError('image can \'t read:' + file_path)
        return image

    def set_gan(self):
        cifar_norm_mean = (0.5, 0.5, 0.5)
        cifar_norm_std = (0.5, 0.5, 0.5)
        method_list = [
            transforms.ToPILImage(),
            transforms.Resize(self.image_shape),
            transforms.ToTensor(),
            transforms.Normalize(cifar_norm_mean, cifar_norm_std),
        ]
        self.datagen = transforms.Compose(method_list)

    def load_data_new(self, file_path):
        photo_path_set = []
        if file_path[-1] != '/' or file_path[-1] != '\\':
            file_path = file_path + '/'
        label_path = file_path + 'Label'
        image_name_list = os.listdir(file_path)
        for image_name in image_name_list:
            if image_name[-3:] != 'PNG':
                continue
            image_label_path = label_path + '/' + image_name.split('.')[0] + '_label.PNG'
            image_path = file_path + '/' + image_name
            photo_path_set.append({
                    "image":image_path,
                    "gt":image_label_path,
                })
            
        self.photo_set = photo_path_set
        self.total_number = len(self.photo_set)
        print("total:", self.total_number)

    def load_data(self, file_path, gt_path):
        # 对应文件夹的地址
        photo_path_set = []
        #check 路径 
        assert self.check_dir(file_path),'{} path not exist'.format(file_path)
        assert self.check_dir(gt_path),'{} path not exist'.format(gt_path)
        if not (file_path[-1] == '/' or  file_path[-1] == '\\'):
            file_path = file_path + '/'
        if not (gt_path[-1] == '/' or  gt_path[-1] == '\\'):
            gt_path = gt_path + '/'

        assert os.listdir(file_path) == os.listdir(gt_path), 'train and gt path dir not equ'

        middle_dir = os.listdir(file_path)
        for dir_index in middle_dir:
            image_dir_path = file_path + dir_index
            gt_dir_path = gt_path + dir_index

            assert os.listdir(image_dir_path) == os.listdir(gt_dir_path), '{} and {} path image not equ'.format(image_dir_path, gt_dir_path)

            middle_image_name_list =  os.listdir(image_dir_path)
            for image_name in middle_image_name_list:
                photo_path_set.append({
                    "image":image_dir_path + '/' + image_name,
                    "gt":gt_dir_path + '/' + image_name,
                })

        self.photo_set = photo_path_set
        self.total_number = len(self.photo_set)
        print("total:", self.total_number)

    def __getitem__(self, index):
        """
        获取对应index的图像，并视情况进行数据增强
        """
        if index >= self.total_number:
            raise StopIteration
        try:
            re_index = index
            if len(self.photo_set) > 0:
                image_src_path, image_dark_path = self.photo_set[re_index]['image'], self.photo_set[re_index]['gt']
                
                imageSrc = self.datagen(self.read_image_data(image_src_path))
                # if len(imageSrc.shape) == 3: # 取默认灰度图像第 1 个通道
                #     imageSrc = imageSrc[0,:,:].unsqueeze(0)
                imageDark = self.read_image_data(image_dark_path)
                imageDark = self.datagen(imageDark)

            return imageSrc, imageDark

        except Exception as e:
            print("发现异常")
            print(e.__class__.__name__)
            print(e)

    def __len__(self):
        return len(self.photo_set)


if __name__ == '__main__':

    batch_size = 32

    train_dataloader = Dataload(r'H:\DATASET\COLORDATA\train\train_frame', r'H:\DATASET\COLORDATA\train_gt\train_gt_frame')

    train_loader = DataLoader(
        dataset=train_dataloader,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,

    )
    src, dark= train_dataloader[2]
    print(src.shape, dark.shape)
    for data in train_loader:
        img, label = data
        print("img shape{}  label shape{}".format(img.shape, label.shape))
