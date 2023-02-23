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
    def __init__(self, file_path, batch_size=1, data_source=None, gray=False, image_shape=(128, 128)
                 , same_matrix=False, num_require=25, data_type='train', mask=False):
        self.total_number = None
        self.file_path = file_path
        self.data_source = data_source
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.data_type = data_type
        self.mask = mask
        self.label_dict = {
            0: 'Bacillariophyta',
            1: 'Chlorella',
            2: 'Chrysophyta',
            3: 'Dunaliella_salina',
            4: 'Platymonas',
            5: 'translating_Symbiodinium',
            6: 'bleaching_Symbiodinium',
            7: 'normal_Symbiodinium}'
        }
        self.num_class = len(self.label_dict)
        self.photo_set = []
        self.same_matrix = same_matrix
        self.gray = gray
        self.num_require = num_require
        self.load_data(file_path)
        self.set_gan()

        # self.X_train, self.Y_train = self.load_all_data(False ,gray, "train")
        # self.X_val, self.Y_val = self.load_all_data(False ,gray, "val")

    def check_dir(self, path):
        if (not os.path.exists(path)):
            return 0
        return 1

    def read_image_data(self, file_path, gray=False):
        if (gray):
            image = cv2.imread(file_path,2)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_shape)

        if (image is None):
            raise RuntimeError('image can \'t read:' + file_path)
        return image

    def set_gan(self):
        cifar_norm_mean = (0.49139968, 0.48215827, 0.44653124)
        cifar_norm_std = (0.24703233, 0.24348505, 0.26158768)
        method_list = [
            transforms.ToPILImage(),
            transforms.Resize(self.image_shape),
            transforms.ToTensor(),
            transforms.Normalize(cifar_norm_mean, cifar_norm_std),
        ]
        self.datagen = transforms.Compose(method_list)

    def load_data(self, file_path):
        file_path = r'E:\Data\Frame\train'
        # 对应文件夹的地址
        photo_path_set = []
        middle = file_path + "\\src\\" + "frames\\"
        label_path = file_path + "\\dark\\" + "frames\\"
        for i in os.listdir(middle):
            src_path = middle + i
            dark_path = label_path + i

            if os.path.isdir(src_path) and os.path.isdir(dark_path):
                photo_path_set.append([src_path, dark_path])
            else:
                print("error")
        for i in photo_path_set:
            src_paths = os.walk(i[0])
            dark_paths = os.walk(i[1])
            for src_root, src_dirs, src_files in src_paths:
                for dark_root, dark_dirs, dark_files in dark_paths:
                    if len(src_dirs) != len(dark_dirs):
                        raise RuntimeError('src and dark files not match')
                    else:
                        for src_file in src_files:
                            for dark_file in dark_files:
                                if src_file == dark_file:
                                    self.photo_set.append(
                                        [os.path.join(src_root, src_file), os.path.join(dark_root, dark_file)])
        self.total_number = len(self.photo_set)
        print("total:", self.total_number)
        # middle = file_path + "\\images\\"
        # label_path = file_path + "\\labels\\"
        #
        # for i in os.listdir(middle):
        #     num = os.path.splitext(i)[0]
        #     x = middle + num + '.png'
        #     y = label_path + num + '.txt'
        #     self.photo_set.append([x,y])
        #
        # if(not self.check_dir(middle)):
        #     if(self.dataset_type == "train"):
        #         raise RuntimeError('train dir not exists:'+ middle)
        #     else:
        #         raise RuntimeError('val dir not exists:' + middle)

    def generate_mask(self, image):
        image_bounding = cv2.Canny(image, 0, 255)
        mask = cv2.dilate(image_bounding, np.ones((3, 3), np.uint8), iterations=5)
        return mask

    def generate_gan_file(self, file_path, save_path):
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        count = 0
        image_path_list = glob.glob(file_path + "*.png")
        for image_path in image_path_list:
            image = cv2.imread(image_path)
            w, h, c = image.shape
            image2 = cv2.resize(image, (224, 224))
            image_masked = self.generate_mask(image2)
            image_masked = cv2.resize(image_masked, (h, w))
            image_masked = cv2.bitwise_and(image, image, mask=image_masked)
            image_name = image_path.split('\\')[-1]
            if count % 100 == 0:
                print("{}/{} \n".format(count, len(image_path_list)))
            # print(save_path + image_name)
            cv2.imwrite(save_path + image_name, image_masked)
            count += 1

    def write_image(self, file_pa):
        pass

    def __getitem__(self, index):
        """
        获取对应index的图像，并视情况进行数据增强
        """
        if index >= self.total_number:
            raise StopIteration
        try:
            re_index = index
            if len(self.photo_set) > 0:
                image_src_path, image_dark_path = self.photo_set[re_index][0], self.photo_set[re_index][1]
                imageSrc = self.read_image_data(image_src_path, gray=False)
                imageDark = self.read_image_data(image_dark_path, gray=True)
                # print(imageSrc.shape)
                # print(imageDark.shape)

            if self.mask:
                image_masked = self.generate_mask(imageSrc)
            # if self.datagen is not None:
            #     image = self.datagen(imageSrc)
            #     if self.mask:
            #         image_masked = self.datagen(image_masked)
            # if self.mask:
            #     return image, imageDark, image_masked
            return imageSrc, imageDark

        except Exception as e:
            print("发现异常")
            print(e.__class__.__name__)
            print(e)

    def __len__(self):
        return len(self.photo_set)


if __name__ == '__main__':

    batch_size = 32

    train_dataloader = Dataload(r"E:\Dataset\training_set\train")

    train_loader = DataLoader(
        dataset=train_dataloader,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,

    )
    # re_index, b= train_loader[500]
    # print("photo")
    # print(re_index)
    # print("label")
    # print(b)
    # print(img1.shape, label1.shape)
    # print(img2.shape, label2.shape)
    for data in train_loader:
        img, label = data
        print("img shape{}  label shape{}".format(img.shape, label.shape))
