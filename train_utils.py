import tqdm
import os
import torch
from torch.utils.tensorboard import SummaryWriter


def train_net(model, trainloader, val_loader, optimizer, scheduler, epoch, device, loss_fn):
    train_losses = []
    val_losses = []
    val_acc = []
    train_acc=[]
    # tensorboard
    writer = SummaryWriter('logs/')

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

            h = model(img)

            loss = loss_fn(h, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = img.size(0)
            running_loss += loss.item()
            total += batch_size

            _, y_pred = h.max(1)
            n_acc += (label == y_pred).float().sum().item()

        scheduler.step()

        # train_dataset loss
        train_losses.append(running_loss / i)

        # train_dataset acc
        train_acc.append(n_acc / total)

        # valid_dataset acc
        val_loss, acc = eval_net(model, val_loader, device, loss_fn)
        val_acc.append(acc)
        val_losses.append(val_loss)
        # epoch
        print(f'epoch: {epoch+1}, train_loss:{round(train_losses[-1], 6)}, valid_loss:{round(val_losses[-1], 6)}, '
              f'train_acc:{round(train_acc[-1],4)},val_acc: {round(val_acc[-1],4)}', flush=True)

        writer.add_scalars("Accuracy", {'train_acc': train_acc[-1], 'val_acc': val_acc[-1]}, epoch)
        writer.add_scalars("Loss", {'train_loss': train_losses[-1], 'val_loss': val_losses[-1]}, epoch)
        writer.add_scalar('Lr', scheduler.optimizer.param_groups[0]['lr'], epoch)

        # model save
        torch.save(model.cpu().state_dict(),'./models/model_'+str(epoch)+'.pth')

    writer.close()

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
