import numpy as np
import torch
from statistics import mean
from main import *
from torch.utils.tensorboard import SummaryWriter


def train_start(_parameters):

    mini_loss = 10000.0

    writer = SummaryWriter('./logs')
    HW1_M.train()
    _loss = _parameters['loss']

    train_loss_arry = []
    train_acc_arry = []
    val_loss_arry = []
    val_acc_arry = []

    for epoch in range(5000):
        loss_epoch = 0
        acc_epoch = []
        for data in train_loador:
            xs ,ys = data
            xs = xs.to(device)
            ys = ys.to(device)

            outputs = HW1_M(xs)
            loss_batch = _loss(outputs,ys)

            # if(epoch%500 == 0):
            #     print('outputs: ')
            #     print(outputs)
            #     print()
            #     print('ys :')
            #     print(ys)

            optim.zero_grad()
            loss_batch.backward()
            optim.step()

            loss_epoch += loss_batch.item()
            acc_epoch.append(1- np.sum(abs(outputs.cpu().data.numpy() - ys.cpu().data.numpy()))/np.sum(ys.cpu().data.numpy()))

        train_loss_arry.append(loss_epoch/ train_set.__len__())
        train_acc_arry.append(mean(acc_epoch))

        writer.add_scalar('train acc',train_acc_arry[-1],epoch)
        writer.add_scalar('train loss',train_loss_arry[-1],epoch)


        if epoch %  5 == 0:
            HW1_M.eval()
            loss_epoch = 0
            acc_epoch = []
            with torch.no_grad():
                for data in val_loador:
                    xs, ys = data
                    xs = xs.to(device)
                    ys = ys.to(device)

                    outputs = HW1_M(xs)
                    loss_batch = _loss(outputs, ys)

                    loss_epoch += loss_batch.item()
                    acc_epoch.append(1- np.sum(abs(outputs.cpu().data.numpy() - ys.cpu().data.numpy()))/np.sum(ys.cpu().data.numpy()))

                val_loss_arry.append(loss_epoch/val_set.__len__())
                val_acc_arry.append(mean(acc_epoch))

                writer.add_scalar('val acc ',val_acc_arry[-1],epoch)
                writer.add_scalar('val loss',val_loss_arry[-1],epoch)

                if val_loss_arry[-1] < mini_loss :
                    torch.save(HW1_M,'./model_save/HW1.path')
                    mini_loss = val_loss_arry[-1]

        if epoch %2 ==0 :
            print('--- epoch: {} --- train acc: {} --- train loss: {} --- val loss: {} --- '.format(epoch,train_acc_arry[-1],train_loss_arry[-1],val_loss_arry[-1])+'val acc: {}'.format(val_acc_arry[-1]))

