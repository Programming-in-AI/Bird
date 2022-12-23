import tqdm
import os
import torch
from torch.utils.tensorboard import SummaryWriter
import platform
from Net import *
import torch.nn as nn

import numpy as np
import _pickle as pickle


def train_net(TotalNet, trainloader, val_loader, optimizer, scheduler, epoch, device, loss_fn, top_k):
    train_losses = []
    val_losses = []
    val_acc = []
    train_acc=[]
    # tensorboard
    #writer = SummaryWriter(str(top_k)+'_logs/')

    # model save path
    os.makedirs('./models/', exist_ok=True)

    for epoch in range(epoch):
        running_loss = 0.0
        # train mode
        TotalNet.train()

        total = 0
        n_acc = 0

        for i, (img, label) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader)):
            TotalNet = TotalNet.to(device)
            img = img.to(device)
            label = label.to(device)

            logit_mixed = TotalNet(img) # h = [batch_size, 200]
            batch_size = img.size(0)

            _, y_pred = logit_mixed.max(1)  # y_pred_c : [batch_size, k ] 하나의 이미지마다 k개의 class가 있음


            # loss
            #loss = loss_fn(TotalNet.logit2, label) + loss_fn(TotalNet.logit1, label)
            loss = loss_fn(TotalNet.logit2, label)
            # loss = loss_fn(logit_mixed, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            total += batch_size

            n_acc += (label == y_pred).float().sum().item()

        scheduler.step()

        # train_dataset loss
        train_losses.append(running_loss / i)

        # train_dataset acc
        train_acc.append(n_acc / total)

        # valid_dataset acc
        val_loss, acc = eval_net(TotalNet, val_loader, device, loss_fn)
        val_acc.append(acc)
        val_losses.append(val_loss)
        # epoch
        print(f'epoch: {epoch+1}, train_loss:{round(train_losses[-1], 6)}, valid_loss:{round(val_losses[-1], 6)}, '
              f'train_acc:{round(train_acc[-1],4)},val_acc: {round(val_acc[-1],4)}', flush=True)

        # writer.add_scalars("Accuracy", {'train_acc': train_acc[-1], 'val_acc': val_acc[-1]}, epoch)
        # writer.add_scalars("Loss", {'train_loss': train_losses[-1], 'val_loss': val_losses[-1]}, epoch)
        # writer.add_scalar('Lr', scheduler.optimizer.param_groups[0]['lr'], epoch)

        # model save
        torch.save(TotalNet.cpu().state_dict(), './models/' + str(top_k) + '_model_'+str(epoch)+'.pth')

    # writer.close()

    return train_losses, val_losses, train_acc, val_acc


def eval_net(model, data_loader, device, loss_fn):
    # Dropout or BatchNorm 没了
    model.eval()
    eval_loss = 0
    ys = []
    ypreds = []

    for i, (x, y) in enumerate(data_loader):
        # send to device
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            h = model(x)
            loss = loss_fn(h, y)
            _, y_pred = h.max(1)

        ys.append(y)
        ypreds.append(y_pred)
        eval_loss += loss.item()

    ys = torch.cat(ys)
    ypreds = torch.cat(ypreds)
    acc = (ys == ypreds).float().sum() / len(ys)
    eval_loss = eval_loss / i

    return eval_loss, acc.item()


# def select_topk_cls(indices, y_pred_c):
#     # indices = [200,4], y_pred_c = [bs, top_k]
#     indices = indices.detach().cpu().numpy()
#     tmp = []
#     # y_pred_c = torch.tensor([[2, 67, 198], [4, 45, 61], ...])
#     for i, item in enumerate(y_pred_c):  # for loop size = bs
#         item = item.cpu().numpy()
#         # print('indices[:item] size:', torch.Tensor(indices[item, :]).size())
#         tmp.append(indices[item, :])
#     tmp = torch.Tensor(np.array(tmp))
#     # print('tmp size: ', tmp.size())
#     topk_cls = torch.cat([tmp], dim=0)  # topk_cls =[bs, top_k, 4]
#
#     return topk_cls


def set_device():
    if platform.system() == 'Darwin':
        device = 'mps'
    elif platform.system() == 'Windows':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        raise Exception("Only available Windows or mac OS")
    print("Device: {}".format(device))

    return device


# def FC(x, out, dropout_rate, device): # x = (bs, 2048, 196)->(bs, 1024, 196)
#     bs, ch, pixel = x.size()
#     x = x.view(-1,ch)
#     fc = nn.Linear(ch, out).to(device)
#     x = fc(x)
#     x = nn.ReLU(inplace=False)(x)
#     x = nn.Dropout(dropout_rate)(x)  # x = (bs * 196, 1024)
#     return x.view(bs, out, pixel)


# def FC2(x, out, device): # x = (bs, 1024)->(bs, 2048)
#     bs, in_dim = x.size()
#     fc = nn.Linear(in_dim, out).to(device)
#     x = fc(x)
#     x = nn.ReLU(inplace=False)(x)  # x = (bs, 2048)
#     return x


def fine_grained_classifier(x, device):
    classes_num = 200
    x = nn.AdaptiveAvgPool2d(output_size=1).to(device)(x)   # x= (bs, ch, H, W)
    x = nn.Dropout(p=0.2).to(device)(x)  # x= (bs, ch, 1, 1)
    x = nn.Conv2d(x.size(1), classes_num, kernel_size=1, bias=False).to(device)(x)  # x= (bs, 200, 1, 1)
    x = x.squeeze()  # x= (bs, 200)
    if x.size(0) == 200:
        x = x.unsqueeze(0)
    return x

