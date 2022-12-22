import tqdm
import os
import torch
from torch.utils.tensorboard import SummaryWriter
import platform
from Net import *
import torch.nn as nn
from ClassnameProcessing import read_glove_vecs, sentences_to_indices, class_embedding
import numpy as np
import _pickle as pickle


def train_net(model, trainloader, val_loader, optimizer, scheduler, epoch, device, loss_fn, top_k):
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
        model.train()

        total = 0
        n_acc = 0

        for i, (img, label) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader)):

            model = model.to(device)
            img = img.to(device)
            label = label.to(device)

            logit1 = model(img) # h = [batch_size, 200]
            #Total_logit = TotalNet(img)
            batch_size = img.size(0)

            _, y_pred_c = torch.topk(logit1, k=top_k, dim=1)  # y_pred_c : [batch_size, k ] 하나의 이미지마다 k개의 class가 있음

            indices, word2index, index2word, word2vec = make_indices()  # indices = [200,4]

            add_emb = np.expand_dims(word2vec[0], 0)
            word2vec = np.append(word2vec, add_emb, axis=0)
            word2vec[0] = np.zeros((word2vec.shape[1]))
            word2vec = torch.Tensor(word2vec)

            tpk_cls = select_topk_cls(indices, y_pred_c)  # tpk_cls =[bs, top_k, 4]


            # ClassEmbedding instance
            CE = ClassEmbedding(tpk_cls, word2vec, emb_dim=300)
            cls_emb = CE(tpk_cls)
            #cls_emb = class_embedding(tpk_cls, word2vec, emb_dim=300)  # cls_emb = [bs, 1024, k]
            cls_emb = cls_emb.to(device)

            # CNN part
            FM_model = torch.nn.Sequential(*(list(model.children())[:-2]))

            # ftm
            FM_model = FM_model.to(device)
            ftm = FM_model(img)  # ftm = [bs, 2048, 14, 14]
            bs, ch, W, H = ftm.size()
            ftm = ftm.view(bs, ch, -1)  # ftm = (bs, 2048, 196)

            # v, cls_emb, att_w
            fc1 = FC(ftm, 1024, dropout_rate=.5, device=device)
            v = fc1(ftm)
            # v = FC(ftm, 1024, dropout_rate=.5, device=device)  # v = (bs, 1024, 196)
            fc2 = FC(cls_emb, 1024, dropout_rate=.8, device=device)
            cls_emb = fc2(cls_emb)
            # cls_emb = FC(cls_emb, 1024, dropout_rate=.8, device=device)  # cls_emb = (bs, 1024, k)
            att_w = torch.einsum('bdv,bdq->bvq', v, cls_emb)  # att_w = (bs, 196, k)
            att_w = nn.Softmax(dim=1)(att_w)  # att_w = (bs, 196, k)

            J_emb = torch.einsum('bdv,bvq,bdq->bd', v, att_w, cls_emb)  # J_emb = (bs, 1024)
            fc3 = FC2(int(ftm.shape[1]), device=device)
            J_emb = fc3(J_emb)
            # J_emb = FC2(J_emb, int(ftm.shape[1]), device=device)  # J_emb = (bs, 2048)
            J_emb = J_emb.unsqueeze(2)  # J_emb = (bs, 2048, 1)
            J_emb = J_emb + ftm  # (bs, 2048, 1) + (bs, 2048, 196)
            # att_w = att_w.sum(dim=2, keepdim=True)  # att_w = （bs, 196, 1）

            J_emb = J_emb.view(bs, ch, H, W)

            FGC = FineGrainedClassifier(J_emb, device=device)
            logit2 = FGC(J_emb)
            # logit2 = fine_grained_classifier(J_emb, device=device)

            _, y_pred_f = logit2.max(1)
            alpha = 0.5
            logit_mixed = alpha * logit1 + (1-alpha) * logit2
            _, y_pred_mixed = logit_mixed.max(1)

            # loss
            loss = loss_fn(logit2, label) + loss_fn(logit1, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            total += batch_size

            n_acc += (label == y_pred_mixed).float().sum().item()

        scheduler.step()

        # train_dataset loss
        train_losses.append(running_loss / i)

        # train_dataset acc
        train_acc.append(n_acc / total)

        # valid_dataset acc
        val_loss, acc = eval_net(model, FM_model, CE, fc1, fc2, fc3, FGC, val_loader, device, loss_fn, top_k)
        val_acc.append(acc)
        val_losses.append(val_loss)
        # epoch
        print(f'epoch: {epoch+1}, train_loss:{round(train_losses[-1], 6)}, valid_loss:{round(val_losses[-1], 6)}, '
              f'train_acc:{round(train_acc[-1],4)},val_acc: {round(val_acc[-1],4)}', flush=True)

        # writer.add_scalars("Accuracy", {'train_acc': train_acc[-1], 'val_acc': val_acc[-1]}, epoch)
        # writer.add_scalars("Loss", {'train_loss': train_losses[-1], 'val_loss': val_losses[-1]}, epoch)
        # writer.add_scalar('Lr', scheduler.optimizer.param_groups[0]['lr'], epoch)

        # model save
        torch.save(model.cpu().state_dict(), './models/' + str(top_k) + '_model_'+str(epoch)+'.pth')

    # writer.close()

    return train_losses, val_losses, train_acc, val_acc


def eval_net(model, FM_model, CE, fc1, fc2, fc3, FGC, data_loader, device, loss_fn, top_k):
    # Dropout or BatchNorm 没了
    model.eval()
    FM_model.eval()
    CE.eval()
    fc1.eval()
    fc2.eval()
    fc3.eval()
    FGC.eval()
    eval_loss = 0
    ys = []
    ypreds = []

    for i, (img, label) in enumerate(data_loader):

        # send to device
        model = model.to(device)
        img = img.to(device)
        label = label.to(device)
        with torch.no_grad():
            logit1 = model(img)  # h = [batch_size, 200]

            batch_size = img.size(0)

            _, y_pred_c = torch.topk(logit1, k=top_k, dim=1)  # y_pred_c : [batch_size, k ] 하나의 이미지마다 k개의 class가 있음

            indices, word2index, index2word, word2vec = make_indices()  # indices = [200,4]

            add_emb = np.expand_dims(word2vec[0], 0)
            word2vec = np.append(word2vec, add_emb, axis=0)
            word2vec[0] = np.zeros((word2vec.shape[1]))
            word2vec = torch.Tensor(word2vec)

            tpk_cls = select_topk_cls(indices, y_pred_c)  # tpk_cls =[bs, top_k, 4]
            cls_emb = CE(tpk_cls)  # cls_emb = [bs, 1024, k]
            cls_emb = cls_emb.to(device)


            # ftm
            FM_model = FM_model.to(device)
            ftm = FM_model(img)  # ftm = [bs, 2048, 14, 14]
            bs, ch, W, H = ftm.size()
            ftm = ftm.view(bs, ch, -1)  # ftm = (bs, 2048, 196)

            # v, cls_emb, att_w
            v = fc1(ftm)  # v = (bs, 1024, 196)
            cls_emb = fc2(cls_emb)  # cls_emb = (bs, 1024, k)
            att_w = torch.einsum('bdv,bdq->bvq', v, cls_emb)  # att_w = (bs, 196, k)
            att_w = nn.Softmax(dim=1)(att_w)  # att_w = (bs, 196, k)

            J_emb = torch.einsum('bdv,bvq,bdq->bd', v, att_w, cls_emb)  # J_emb = (bs, 1024)
            J_emb = fc3(J_emb)  # J_emb = (bs, 2048)
            J_emb = J_emb.unsqueeze(2)  # J_emb = (bs, 2048, 1)
            J_emb = J_emb + ftm  # (bs, 2048, 1) + (bs, 2048, 196)
            # att_w = att_w.sum(dim=2, keepdim=True)  # att_w = （bs, 196, 1）

            J_emb = J_emb.view(bs, ch, H, W)

            logit2 = FGC(J_emb)

            _, y_pred_f = logit2.max(1)
            alpha = 0.5
            logit_mixed = alpha * logit1 + (1 - alpha) * logit2
            _, y_pred_mixed = logit_mixed.max(1)

            # loss
            loss = loss_fn(logit2, label) + loss_fn(logit1, label)

    ###
        ys.append(label)
        ypreds.append(y_pred_mixed)
        eval_loss += loss.item()

    ys = torch.cat(ys)
    ypreds = torch.cat(ypreds)
    acc = (ys == ypreds).float().sum() / len(ys)
    eval_loss = eval_loss / i

    return eval_loss, acc.item()


def select_topk_cls(indices, y_pred_c):
    # indices = [200,4], y_pred_c = [bs, top_k]
    indices = indices.detach().cpu().numpy()
    tmp = []
    # y_pred_c = torch.tensor([[2, 67, 198], [4, 45, 61], ...])
    for i, item in enumerate(y_pred_c):  # for loop size = bs
        item = item.cpu().numpy()
        # print('indices[:item] size:', torch.Tensor(indices[item, :]).size())
        tmp.append(indices[item, :])
    tmp = torch.Tensor(np.array(tmp))
    # print('tmp size: ', tmp.size())
    topk_cls = torch.cat([tmp], dim=0)  # topk_cls =[bs, top_k, 4]

    return topk_cls


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


def make_indices():
    with open('./data/Bird_classes.pkl', 'rb') as f:
        classes = np.array(pickle.load(f))

    word2index, index2word, word2vec = read_glove_vecs(
        './data/Bird_glove6b_init_300d.npy', './data/Bird_dictionary.pkl')
    # print(word2vec.shape)

    word2index[index2word[0]] = len(word2index)
    word2vec = torch.tensor(word2vec, dtype=torch.float)
    # print(word2vec.size())

    indices = sentences_to_indices(classes, word2index, 4)
    indices = torch.tensor(indices)

    return indices, word2index, index2word, word2vec