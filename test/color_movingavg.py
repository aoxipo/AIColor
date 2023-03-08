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

def MoveSmoothBox(now_frame, frame_box, alpha_origin = 0.1):
    if len(frame_box) == 0:
        return now_frame
    alpha = (1 - alpha_origin) / (len(frame_box))
    now_frame = alpha_origin * now_frame
    for frame in frame_box: 
        now_frame += alpha * frame
    return now_frame

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=".", help="path of input videos")
    parser.add_argument("--output", type=str, default="frames", help="path of output clips")
    parser.add_argument("--alpha", type=float, default=0.5, help="origin frame rate")
    parser.add_argument("--len", type=int, default=2, help="box frame len")
    opt = parser.parse_args()

    image_path = opt.input
    save_path = opt.output
    alpha = opt.alpha
    box_len = opt.len
    
    for (root, dirs, files) in os.walk(image_path):
        now_dir = root.split('\\')[-1]
        frame_box = []
        for index_file in files:
            if index_file[-3:] != 'png':
                break
            image_path = root + '/' + index_file
            img = cv2.imread(image_path)
            img = np.array(img, dtype = np.uint16)

            frame_box.append(img)
            if len(frame_box) >= box_len:
                frame_box.pop(0)
            
            img_new = MoveSmoothBox( img, frame_box, alpha)
            img_new = np.array(img_new, dtype=np.uint8)
  
            
            image_path_dir = '{}/{}'.format(save_path,now_dir)
            if not os.path.exists(image_path_dir):
                os.mkdir(image_path_dir)
            image_path_saved = '{}/{}'.format(image_path_dir, index_file)
            cv2.imwrite(image_path_saved, img_new)
            print(image_path_saved)


