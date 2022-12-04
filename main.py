from dataloader import CustomDataset, Dataloader
from train import train_net
from Net import Net
import torch
import platform
import timm
import torchvision.models as models
from dataloader import CustomDataset
from vit_pytorch import ViT

def main():
    root_dir = './CUB_200_2011/images/'

    resize_factor = [600,600]
    print('[Dataset Processing...]')
    Dataset = CustomDataset(root_dir, isTrain=True)
    print("Training data size : {}".format(Dataset.__len__()[0]))
    print("Validating data size : {}".format(Dataset.__len__()[1]))
    batch_size = 8
    train_dataloader = Dataloader(Dataset.train_dataset, batch_size)
    val_dataloader = Dataloader(Dataset.val_dataset, batch_size)




    # model = Net()
    torch.manual_seed(42)
    num_classes = 200

    # Resnet
    model = models.resnet101(weights='ResNet101_Weights.DEFAULT')
    fc_input_dim = model.fc.in_features
    model.fc = torch.nn.Linear(fc_input_dim, num_classes)

    # viT
    # model = ViT(
    #     image_size=448,
    #     patch_size=32,
    #     num_classes=num_classes,
    #     dim=1024,
    #     depth=6,
    #     heads=16,
    #     mlp_dim=2048,
    #     dropout=0.1,
    #     emb_dropout=0.1
    # )

    if platform.system() == 'Darwin':
        device = 'mps'
    elif platform.system() == 'Windows':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: {}".format(device))

    epoch = 10
    learning_rate = 0.0001
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.1)
    train_net(model, train_dataloader, val_dataloader, optimizer, epoch, device, loss_function)


if __name__ == '__main__':
    main()
