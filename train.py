from dataloader import CustomDataset, Dataloader
import platform
import torch
from train_utils import train_net
import torchvision.models as models
# from vit_pytorch import ViT
# import timm


def train(root_dir, device):

    print('[Dataset Processing...]')
    dataset = CustomDataset(root_dir, isTrain=True)

    print("Training data size : {}".format(dataset.__len__()[0]))
    print("Validating data size : {}".format(dataset.__len__()[1]))
    batch_size = 2
    train_dataloader = Dataloader(dataset.train_dataset, batch_size)
    val_dataloader = Dataloader(dataset.val_dataset, batch_size)

    # model = Net()

    num_classes = 200

    # Resnet
    model = models.resnet101(weights='ResNet101_Weights.DEFAULT')
    fc_input_dim = model.fc.in_features
    model.fc = torch.nn.Linear(fc_input_dim, num_classes)

    # VIT
    # model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)

    epoch = 10
    learning_rate = 0.001
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[3, 5, 7, 9], gamma=0.5)
    train_net(model, train_dataloader, val_dataloader, optimizer, scheduler, epoch, device, loss_function, top_k = 3)
    # top_k = 3