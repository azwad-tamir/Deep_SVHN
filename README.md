# Deep_SVHN: Numerical Digit Detection and Classification on SVHN dataset
A numerical digit detection system has been build based on deep convolutional neural networks.
The model is trained and tested on the SVHN dataset which consists of bulk multi-digit images
of house numbers. The dataset contains two types of images. The type which consists of raw
uncropped house number images has been chosen. The model consists of two parts; a detector
and a classifier. The raw images are fed to the detector which creates bounding boxes around
each of the separate digits of an image and crops the individual images of the digits. Next, the
individual digit images are fed to the classifier, which classifies the images into 10 classes
starting from '0' to '9'. The detector is based on resnet50 [1] and Yolo-v2 [2-3]. It is built from
scratch using the PyTorch machine learning framework. The individual accuracy of the detector
and the classifier has been evaluated. The detector reports a training accuracy of 91% and a test
accuracy of 59% while the classifier reports a training accuracy of 94% and a test accuracy of
92.11%. The overall training and testing accuracy of the entire system is found to be 86% and
54.41% respectively

Run Instructions:
1. Download the model weights and put them in the main folder: 
   saved_model.pth: https://knightsucfedu39751-my.sharepoint.com/:u:/g/personal/a_tamir_knights_ucf_edu/ET42pMxfe39OiIou9wkrHH0Bwu3cpJdeHM0I43O-5m422g?e=Wa7DRV
   weights.h5: https://knightsucfedu39751-my.sharepoint.com/:u:/g/personal/a_tamir_knights_ucf_edu/EUom-5PxTdZPv2Duaob--lEB9BObF1c0iP54rMfKQjcW7w?e=3nrroF

2. Run - "pip install -r requirements.txt"

3. Then run "main.py"

Optional- If you want to run detector and classifier, you need the data downloaded from the SVHN dataset and store it in "data" folder.
