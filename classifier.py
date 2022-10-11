import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
from bokeh.plotting import figure
from bokeh.io import show
from bokeh.models import LinearAxis, Range1d

from torch.optim.lr_scheduler import StepLR
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix

result_path = './logs'

# Hyperparameters
num_epochs = 300
num_classes = 20
batch_size = 100
learning_rate = 0.001

# Defining various directories
DATA_PATH = './data'


def get_loader(svhn_path, image_size=32*32, batch_size=100):

    """Builds and returns Dataloader for SVHN dataset."""
    transform = transforms.Compose([
        #transforms.Scale(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    svhn_train = torchvision.datasets.SVHN(root=svhn_path, download=True, transform=transform, split='train')

    svhn_test = torchvision.datasets.SVHN(root=svhn_path, download=True, transform=transform, split='test')

    svhn_train_loader = torch.utils.data.DataLoader(dataset=svhn_train,
                                              batch_size=batch_size,
                                              shuffle=True)

    svhn_test_loader = torch.utils.data.DataLoader(dataset=svhn_test,
                                                   batch_size=batch_size,
                                                   shuffle=False)

    return svhn_train, svhn_test, svhn_train_loader, svhn_test_loader

train_data, test_data, train_loader1, test_loader = get_loader(DATA_PATH, batch_size)

train_labels = train_data.labels
test_labels = test_data.labels


#This class creates a Pytorch dataset from the python array
class MyDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).long()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)


## Get the training images(cropped from original images)
filelist_train = glob.glob('./train_images/*.png')

def sortKeyFunc(s):
    return int(os.path.basename(s)[:-4])

filelist_train.sort(key=sortKeyFunc)
train_array = np.array([np.array(Image.open(fname)) for fname in filelist_train])
train_array = train_array.swapaxes(1,3)
train_array = train_array.swapaxes(2,3)

## Printing the size to confirm
print(train_labels.shape)
print(train_array.shape)


## Get the test images(cropped from original images)
filelist_test = glob.glob('./test_images/*.png')

def sortKeyFunc(s):
    return int(os.path.basename(s)[:-4])

filelist_test.sort(key=sortKeyFunc)
test_array= np.array([np.array(Image.open(fname)) for fname in filelist_test])
test_array = test_array.swapaxes(1,3)
test_array = test_array.swapaxes(2,3)


## Printing the size to confirm
print(test_labels.shape)
print(test_array.shape)



## Make the dataset
train_dataset = MyDataset(train_array, train_labels)
test_dataset = MyDataset(test_array, test_labels)

# Data loader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader =  DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
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

USE_GPU = True
model = ConvNet()
print(model)
if USE_GPU:
    model.cuda()

## name of the architecture
architecture = '{}-{}'.format(model, 'train')


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=10000, gamma=0.5)


# Train the model
total_step = len(train_loader)
loss_list = []
acc_list = []



## Training Stage
train_pred = torch.zeros((train_array.shape[0]))
cudnn.benchmark = True
for epoch in range(num_epochs):
    index = 0
    for i, (images, labels) in enumerate(train_loader):

        images = images.cuda()
        outputs = model(images)
        loss = criterion(outputs, labels.cuda())
        loss_list.append(loss.item())

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Track the accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels.cuda()).sum().item()
        acc_list.append(correct / total)

        train_pred[(batch_size * index):(batch_size * index + batch_size)] = predicted
        index += 1

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))

        ####### MODEL SAVE ######
    if (epoch+1) % 10 == 0:
        save_file_path = os.path.join(result_path,
                                      'save_{}.pth'.format(epoch))
        states = {
            'epochs': epoch,
            'arch': architecture,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        torch.save(states, save_file_path)

######### MODEL LOAD #######
resume_from_model = True
model_path = './logs/save_29.pth'
test_pred = torch.zeros((test_array.shape[0]))
if resume_from_model:
    print('loading checkpoint {}'.format(model_path))
    checkpoint = torch.load(model_path)
    assert architecture == checkpoint['arch']

    begin_epoch = checkpoint['epochs']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


#Test the model
index =0
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images.cuda())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.cuda().size(0)
        correct += (predicted == labels.cuda()).sum().item()

        test_pred[(batch_size * index):(batch_size * index + batch_size)] = predicted
        index += 1


    print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))

# Save the model and plot the loss and accuracy
p = figure(y_axis_label='Loss', width=850, y_range=(0, 1), title='PyTorch ConvNet results')
p.extra_y_ranges = {'Accuracy': Range1d(start=0, end=100)}
p.add_layout(LinearAxis(y_range_name='Accuracy', axis_label='Accuracy (%)'), 'right')
p.line(np.arange(len(loss_list)), loss_list)
p.line(np.arange(len(loss_list)), np.array(acc_list) * 100, y_range_name='Accuracy', color='red')
show(p)


# Creating and printing the confusion matrix for the training and testing datasets respectively:
confusion_mat_train = np.zeros((10,10))
confusion_mat_test = np.zeros((10,10))

confusion_mat_train = confusion_matrix(train_pred, train_labels)
confusion_mat_test = confusion_matrix(test_pred, test_labels)
print('\nConfusion matrix for the train and test set are: \n', confusion_mat_train, '\n\n\n', confusion_mat_test)