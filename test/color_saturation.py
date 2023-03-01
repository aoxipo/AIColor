import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def PSAlgorithm(rgb_img, increment):
    img = rgb_img * 1.0
    img_min = img.min(axis=2)
    img_max = img.max(axis=2)
    img_out = img
    delta = (img_max - img_min) / 255.0
    value = (img_max + img_min) / 255.0
    L = value/2.0
    mask_1 = L < 0.5
    if value == 0 :
        s1 = 0.5
    else:
        s1 = delta/(value)
    if value == 2:
        s2 = 0.5
    else:
        s2 = delta/(2 - value)
    s = s1 * mask_1 + s2 * (1 - mask_1)
    if increment >= 0:
        temp = increment + s
        mask_2 = temp > 1
        alpha_1 = s
        alpha_2 = s * 0 + 1 - increment
        alpha = alpha_1 * mask_2 + alpha_2 * (1 - mask_2)
        alpha = 1/alpha -1
        img_out[:, :, 0] = img[:, :, 0] + (img[:, :, 0] - L * 255.0) * alpha
        img_out[:, :, 1] = img[:, :, 1] + (img[:, :, 1] - L * 255.0) * alpha
        img_out[:, :, 2] = img[:, :, 2] + (img[:, :, 2] - L * 255.0) * alpha
    else:
        alpha = increment
        img_out[:, :, 0] = img[:, :, 0] + (img[:, :, 0] - L * 255.0) * alpha
        img_out[:, :, 1] = img[:, :, 1] + (img[:, :, 1] - L * 255.0) * alpha
        img_out[:, :, 2] = img[:, :, 2] + (img[:, :, 2] - L * 255.0) * alpha
    img_out = img_out/255.0
    # RGB颜色上下限处理(小于0取0，大于1取1)
    mask_3 = img_out < 0
    mask_4 = img_out > 1
    img_out = img_out * (1-mask_3)
    img_out = img_out * (1-mask_4) + mask_4
    return img_out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=".", help="path of input videos")
    parser.add_argument("--output", type=str, default="frames", help="path of output clips")
    parser.add_argument("--increment", type=int, default=0.5, help="increment")
    parser.add_argument("--Inc", type=int, default=0.5, help="increment")
    opt = parser.parse_args()

    image_path = opt.input
    save_path = opt.output
    increment = opt.Inc

    for (root, dirs, files) in os.walk(image_path):
        now_dir = root.split('\\')[-1]
        for index_file in files:
            if index_file[-3:] != 'png':
                break

            image_path = root + '/' + index_file
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img_new = PSAlgorithm(img, increment)
            img_new = cv2.cvtColor( np.array(img_new * 255, dtype=np.uint8), cv2.COLOR_BGR2RGB)
            image_path_dir = '{}{}'.format(save_path,now_dir)
            if not os.path.exists(image_path_dir):
                os.mkdir(image_path_dir)
            image_path_saved = '{}/{}'.format(image_path_dir, index_file)
            cv2.imwrite(image_path_saved, img_new)
            print(image_path_saved)
