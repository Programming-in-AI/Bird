import tqdm
import os
import torch

def train_net(model, trainloader, val_loader, optimizer, epoch, device, loss_fn):
    train_losses = []
    val_acc = []
    train_acc=[]

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
        train_losses.append(running_loss / i)

        # train_dataset acc
        train_acc.append(n_acc / total)

        # valid_dataset acc
        val_acc.append(eval_net(model, val_loader, device))
        # epoch
        print(f'epoch: {epoch+1}, train_loss:{train_losses[-1]}, train_acc:{train_acc[-1]},val_acc: {val_acc[-1]}', flush=True)

        #model save
        if epoch % 3 == 0 and epoch > 10:
            torch.save(model.cpu().state_dict(),'./models/model_'+str(epoch)+'.pth')

    return train_losses, train_acc, val_acc


def eval_net(model, data_loader, device):
    # Dropout or BatchNorm 没了
    model.eval()
    ys = []
    ypreds = []
    for x, y in data_loader:
        # send to device
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            _, y_pred = model(x).max(1)
        ys.append(y)
        ypreds.append(y_pred)


    ys = torch.cat(ys)
    ypreds = torch.cat(ypreds)

    acc = (ys == ypreds).float().sum() / len(ys)
    return acc.item()



