# -*- coding: utf-8 -*-
import json
import cv2
from yolo.backend.utils.box import draw_scaled_boxes
import os
import yolo
import matplotlib.pyplot as plt
from yolo.frontend import create_yolo
from PIL import Image
import numpy as np
import torch
import torch.nn as nn


# 1. create yolo instance
yolo_detector = create_yolo("ResNet50", ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], 416)


## See the image files
img_files = "./1.png"
img = cv2.imread(img_files)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer4 = nn.Sequential(
            nn.Conv2d(192, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()

        self.fc1 = nn.Sequential(
            nn.Linear(4*4*256, 1024),
            nn.ReLU())

        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

## Load the model
model = ConvNet()
checkpoint = torch.load('saved_model.pth')
model.load_state_dict(checkpoint['state_dict'])


# 4. Predict digit region
THRESHOLD = 0.25
classifier = []
DEFAULT_WEIGHT_FILE =  "weights.h5"
yolo_detector.load_weights(DEFAULT_WEIGHT_FILE)
img = cv2.imread(img_files)
boxes, probs = yolo_detector.predict(img, THRESHOLD)

# 4. save detection result
image = draw_scaled_boxes(img,
                          boxes,
                          probs,
                          ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])

print("Total {}-boxes are detected.".format(len(boxes)))
plt.imshow(image)
plt.show()


## Feed the image to the model
for k in range(len(boxes)):
    image = Image.open(img_files)
    cropped_left, cropped_top, cropped_right, cropped_bot = boxes[k]
    image = image.crop(
        [cropped_left - 0.05 * cropped_left, cropped_top - 0.05 * cropped_top, cropped_right + 0.05 * cropped_right,
         cropped_bot + 0.05 * cropped_bot])
    image = image.resize([32, 32])
    image.save("test_%d.png" % (k))
    image = np.array(Image.open("test_%d.png" % (k)), dtype=np.single).reshape(1, 32,32,3)
    image = image.swapaxes(1,3)
    image = image.swapaxes(2,3)
    image = torch.from_numpy(image)

    result = model(image)
    _, predicted = torch.max(result.data, 1)
    predicted = predicted.detach().numpy()
    classifier.append(predicted)

print("Predicted Digit:", classifier[0], classifier[1])
