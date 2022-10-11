# -*- coding: utf-8 -*-
import cv2
from yolo.backend.utils.box import draw_scaled_boxes
import os
import yolo
import glob
from yolo.frontend import create_yolo
import h5py
from PIL import Image
import numpy as np
import tensorflow as tf


def get_attrs(digit_struct_mat_file, index):
    """
    Returns a dictionary which contains keys: label, left, top, width and height, each key has multiple values.
    """
    attrs = {}
    f = digit_struct_mat_file
    item = f['digitStruct']['bbox'][index].item()
    for key in ['label', 'left', 'top', 'width', 'height']:
        attr = f[item][key]
        values = [f[attr[i].item()][0][0]
                  for i in range(len(attr))] if len(attr) > 1 else [attr[0][0]]
        attrs[key] = values
    return attrs


# ##create yolo instance
yolo_detector = create_yolo("ResNet50", ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], 416)

#### load pretrained weighted file
DEFAULT_WEIGHT_FILE =  "weights.h5"
yolo_detector.load_weights(DEFAULT_WEIGHT_FILE)



# #### Load images
example_pointer = 0
index1  = 0
miss = 0
desired_size=400
THRESHOLD = 0.25

def sortKeyFunc(s):
    return int(os.path.basename(s)[:-4])

path_to_dataset_dir = '/home/ryota/Yolo_Project/data/train'
path_to_image_files = glob.glob('/home/ryota/Yolo_Project/data/train/*.png')
path_to_image_files.sort(key=sortKeyFunc)
total_files = len(path_to_image_files)


path_to_box_npy_file = '/home/ryota/Yolo_Project/missed_indices_train_2.npy'
path_to_digit_struct_mat_file = os.path.join(path_to_dataset_dir, 'digitStruct.mat')
digit_struct_mat_file = h5py.File(path_to_digit_struct_mat_file, 'r')
data =[]


# 4. Predict digit region
for i in range(total_files):
    path_to_image_file = path_to_image_files[example_pointer]
    index = int(path_to_image_file.split('/')[-1].split('.')[0]) - 1
    example_pointer += 1
    attrs = get_attrs(digit_struct_mat_file, index)
    label_of_digits = attrs['label']
    length = len(label_of_digits)

    if index %1000 == 0:
        print(index)

    image = cv2.imread(path_to_image_file)
    boxes, probs = yolo_detector.predict(image, THRESHOLD)

    ## If there was no detection
    if len(boxes) ==0:
        for k in range( length):
            image = Image.open(path_to_image_file)
            image = image.resize([32, 32])
            image.save("/home/ryota/Yolo_Project/train_images/%04d.png"%(index1))
            index1 +=1
            data.append(index1)
            miss += 1
        print("Missed")
        continue

    ## Sorting
    prob = np.amax(probs, axis = 1)
    index_new = np.argsort(prob)
    box_new = boxes[index_new]

    if len(prob)>length:
        for k in range(len(prob)-length):
            box_new = np.delete(box_new, 0, 0)

    column = box_new[:,0]
    index_col = np.argsort(column)
    box_new = box_new[index_col]


    # 4. save detection result
    boxes_scaled = box_new
    N = boxes_scaled.shape[0]

    if N== length:
        for k in range(boxes_scaled.shape[0]):
            image = Image.open(path_to_image_file)
            cropped_left, cropped_top, cropped_right, cropped_bot = boxes_scaled[k]
            image = image.crop([cropped_left-0.05*cropped_left, cropped_top-0.05*cropped_top, cropped_right+0.05*cropped_right, cropped_bot+0.05*cropped_bot])
            image = image.resize([32, 32])
            image.save("/home/ryota/Yolo_Project/train_images/%04d.png"%(index1))
            index1 += 1

    elif N < length:
        for k in range(boxes_scaled.shape[0]):
            image = Image.open(path_to_image_file)
            cropped_left, cropped_top, cropped_right, cropped_bot = boxes_scaled[k]
            image = image.crop([cropped_left-0.05*cropped_left, cropped_top-0.05*cropped_top, cropped_right+0.05*cropped_right, cropped_bot+0.05*cropped_bot])
            image = image.resize([32, 32])
            image.save("/home/ryota/Yolo_Project/train_images/%04d.png"%(index1))
            index1 += 1
        for k in range(N, length):
            image = Image.open(path_to_image_file)
            cropped_left, cropped_top, cropped_right, cropped_bot = boxes_scaled[N-1]
            image = image.crop([cropped_left-0.05*cropped_left, cropped_top-0.05*cropped_top, cropped_right+0.05*cropped_right, cropped_bot+0.05*cropped_bot])
            image = image.resize([32, 32])
            image.save("/home/ryota/Yolo_Project/train_images/%04d.png"%(index1))
            index1 += 1

np.save(path_to_box_npy_file, data)
