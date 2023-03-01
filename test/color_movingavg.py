import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def MoveSmooth(now_frame, before_frame = None, alpha = 0.5, beta = 0.5):
    if before_frame is None:
        return now_frame
    return alpha * now_frame + before_frame * beta

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=".", help="path of input videos")
    parser.add_argument("--output", type=str, default="frames", help="path of output clips")

    opt = parser.parse_args()

    image_path = opt.input
    save_path = opt.output

    for (root, dirs, files) in os.walk(image_path):
        now_dir = root.split('\\')[-1]
        index = 0
        for index_file in files:
            if index_file[-3:] != 'png':
                break
            image_path = root + '/' + index_file
            img = cv2.imread(image_path)
            img = np.array(img, dtype = np.uint16)
            if index:
                img_new = MoveSmooth(img, old_img)
            else:
                img_new = MoveSmooth(img)

            img_new = np.array(img_new, dtype=np.uint8)

            old_img = img_new
            
            image_path_dir = '{}{}'.format(save_path,now_dir)
            if not os.path.exists(image_path_dir):
                os.mkdir(image_path_dir)
            image_path_saved = '{}/{}'.format(image_path_dir, index_file)
            cv2.imwrite(image_path_saved, img_new)
            print(image_path_saved)
            index += 1

