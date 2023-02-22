from dataloader import Dataload
from utils.plot import plot_rect_old
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
import cv2

data_path = r"H:\DATASET\CLASSDATA\training_set\train"
out_path = r"H:\DATASET\CLASSDATA\training_set\train\images_mask/"
train_dataloader = Dataload(data_path,same_matrix=False, image_shape=(256,256))
train_dataloader.generate_gan_file(data_path+"\images/", out_path)