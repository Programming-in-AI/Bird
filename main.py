from dataloader import CustomDataset, Dataloader
from train import train_net
from Net import Net
import torch
import platform
import tqdm
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import torchvision
from torchvision.transforms import InterpolationMode


def main():
    root_dir = './CUB_200_2011/images/'

    resize_factor = [256,256]
    print('[Dataset Processing...]')
    Dataset = CustomDataset(root_dir, resize_factor)
    print("Training data size : {}".format(len(Dataset.train_dataset)))
    print("Validating data size : {}".format(len(Dataset.val_dataset)))
    batch_size = 8
    train_dataloader = Dataloader(Dataset.train_dataset, batch_size)
    val_dataloader = Dataloader(Dataset.val_dataset, batch_size)


    #model = Net()
    model = models.resnet50(pretrained=True)
    torch.manual_seed(42)
    fc_input_dim = model.fc.in_features
    model.fc = torch.nn.Linear(fc_input_dim, 200)

    if platform.system() == 'Darwin':
        device = 'mps'
    elif platform.system() == 'Windows':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: {}".format(device))

    epoch = 10
    learning_rate = 0.01
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.1)
    train_net(model, train_dataloader, val_dataloader, optimizer, epoch, device, loss_function)

if __name__ == '__main__':
    main()
