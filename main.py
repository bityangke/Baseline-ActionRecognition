import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torchvision import transforms
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from src.functions import *
from src import transforms_own
from model import model_resnext

from src.dataset_own import Dataset_own


if __name__ == '__main__':



    """######### (0) Initialization #########"""
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    data_path = './'
    summary_path = './'
    batch_size = 32




    """######### (1) Dataset Preparation #########"""
    train_transforms = transforms.Compose([transforms_own.CenterCrop(224),transforms_own.Scale(112)])
    train_set = Dataset_own(root=data_path, transform=train_transforms)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=10, pin_memory=True)

    val_transforms = transforms.Compose([transforms_own.CenterCrop(224),transforms_own.Scale(112)])
    val_set = Dataset_own(root=data_path, transform=val_transforms)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=10, pin_memory=True)
    print("train set: {} | validation set: {}".format(train_set.__len__(), val_set.__len__()))




    """######### (2) Build Model #########"""
    model = model_resnext.resnext50(sample_size=112, sample_duration=16).cuda()
    summary(model, (3,16,112,112))
    model.train()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    train_logger = Logger(os.path.join(summary_path, 'tr.log'),['epoch','loss','acc','lr'])
    val_logger = Logger(os.path.join(summary_path, 'val.log'),['epoch','loss','acc','lr'])





    """######### (3) Train Model #########"""
    for epoch in range(100):  # loop over the dataset multiple times
        loss_avg, acc_avg,  loss_val_avg, acc_val_avg = 0.0, 0.0,  0.0, 0.0
        print_step = 10
        start_time = time.time()

        # train phase for each epoch
        for i, data in enumerate(train_loader):

            inputs, labels = data[0].cuda(), data[1].cuda()  # get the inputs; data is list:[inputs, labels]
            optimizer.zero_grad()  # init params.grad
            with torch.enable_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()  # calc and accumulates params.grad
                optimizer.step()  # update params
            loss_avg += loss.item()
            acc_avg += calc_accuracy(outputs, labels, print_pred=False)

            if i % print_step == 0 and i != 0:
                # print every 10 step
                print_format = '[%d | %d step]  loss: %.4f  |  acc: %.4f  |  time per step: %.2f sec'
                print(print_format%(epoch+1, i, loss_avg/print_step, acc_avg/print_step, (time.time()-start_time)/print_step))
                # log every 10 step
                train_logger.log({'epoch': epoch,'loss': loss_avg/print_step,'acc': acc_avg/print_step, 'lr':optimizer.param_groups[0]['lr']})
                # reset every 10 step
                loss_avg, acc_avg = 0.0, 0.0
                start_time = time.time()

        # validation phase for each epoch
        for j, data_val in enumerate(val_loader, 0):
            inputs, labels = data_val[0].cuda(), data_val[1].cuda()  # get the inputs; data is list:[inputs, labels]
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            acc_val_avg += calc_accuracy(outputs, labels, print_pred=False)
            loss_val_avg += loss.item()
        acc_val_avg = acc_val_avg/int(val_set.__len__())
        loss_val_avg = loss_val_avg/int(val_set.__len__())
        val_logger.log({'epoch': epoch, 'loss': loss_val_avg, 'acc': acc_val_avg, 'lr': optimizer.param_groups[0]['lr']})

        # lr schedule
        scheduler.step(loss_val_avg)
        states = {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),}
        save_path = os.path.join(summary_path, '{}_ep_{}').format(model.name, epoch)
        torch.save(states, save_path)
