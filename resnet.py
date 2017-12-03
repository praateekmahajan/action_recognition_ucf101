from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
import time
from data_loader import DataClass

use_gpu = True and torch.cuda.is_available()
FOLDER_DATASET = "../data/"
IMAGE_DATASET = "UCF101_images/"

dataloader = {'train' : DataClass(FOLDER_DATASET, IMAGE_DATASET, "train1.txt"),
			  'validation' : DataClass(FOLDER_DATASET, IMAGE_DATASET, "validation1.txt")}


class CNNGRU(nn.Module):
    def __init__(self):
        super(CNNGRU, self).__init__()
        self.input_dim = 1000
        self.hidden_layers = 512
        self.rnn_layers = 2
        self.classes = 101
        #         self.sample_rate = 12

        self.conv = torchvision.models.resnet18(pretrained=True)
        for param in self.conv.parameters():
            param.requires_grad = False

        self.lstm = nn.LSTM(self.input_dim, self.hidden_layers, self.rnn_layers)
        self.gru = nn.GRU(self.input_dim, self.hidden_layers, self.rnn_layers, dropout=0.2)

        self.linear = nn.Linear(
         in_features=self.hidden_layers, out_features=self.classes)

    def forward(self, x):
        n, t = x.size(0), x.size(1)
        x = x.view(t * n, x.size(2), x.size(3), x.size(4))
        conv_output = self.conv(x)  # convolve allframes
        conv_output = conv_output.view(n, t, -1).transpose(1, 0)
        out, _ = self.gru(conv_output)  # pass convolution to gru
        lstm_output = out[-1, :, :]
        #         print(lstm_output.size())
        output = self.linear(lstm_output) #linear layer
        return output


model_ft = CNNGRU()
if use_gpu:
    model_ft = model_ft.cuda()

criterion = nn.CrossEntropyLoss()

# Remove all parameters not to be optimized
ignored_params = list(map(id, model_ft.conv.parameters()))
base_params = filter(lambda p: id(p) not in ignored_params,
                     model_ft.parameters())

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD([{'params': base_params}], lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


def train_model(model, criterion, optimizer, scheduler, dataloader, batch_size, use_gpu, num_epochs=25):
    since = time.time()
    dataset_sizes = {x: len(dataloader[x]) for x in ['train', 'validation']}
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            start = time.time()
            # Iterate over data.
            for i in range(int(dataset_sizes[phase] / batch_size)):
                # get the inputs
                inputs, labels = dataloader[phase].getbatch(batch_size)

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(torch.from_numpy(inputs).float().cuda())
                    labels = Variable(torch.from_numpy(labels).cuda())
                else:
                    inputs, labels = Variable(torch.from_numpy(inputs).float()), Variable(torch.from_numpy(labels))

                # zero the parameter gradients
                optimizer.zero_grad()
                if i % 100 == 99:
                    print('{:.0f} videos in {:.0f}m {:.0f}s'.format(100 * float(batch_size),
                                                                    (time.time() - start) // 60,
                                                                    (time.time() - start) % 60))
                    start = time.time()
                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                #                 print(outputs.view(-1), labels.view(1))
                loss = criterion(outputs, labels)
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloader, 32, use_gpu, num_epochs=25)