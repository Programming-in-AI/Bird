from dataloader import CustomDataset, Dataloader
import torch
from train_utils import train_net
import torchvision.models as models
import Net
# from vit_pytorch import ViT
# import timm


def train(root_dir, device, top_k):

    print('[Dataset Processing...]')
    dataset = CustomDataset(root_dir, isTrain=True)

    print("Training data size : {}".format(dataset.__len__()[0]))
    print("Validating data size : {}".format(dataset.__len__()[1]))
    batch_size = 3
    train_dataloader = Dataloader(dataset.train_dataset, batch_size)
    val_dataloader = Dataloader(dataset.val_dataset, batch_size)

    # VIT
    # model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)

    num_classes = 200

    # Resnet
    model = models.resnet101(weights='ResNet101_Weights.DEFAULT')
    fc_input_dim = model.fc.in_features
    model.fc = torch.nn.Linear(fc_input_dim, num_classes)

    # CNN part
    FM_model = torch.nn.Sequential(*(list(model.children())[:-2]))

    # TotalNet
    TotalNet = Net.TotalNet(model, FM_model, top_k=3, device=device)

    epoch = 10
    learning_rate = 0.0001
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[3, 5, 6, 7,8, 9], gamma=0.5)
    train_net(TotalNet, train_dataloader, val_dataloader, optimizer, scheduler, epoch, device, loss_function, top_k = top_k)
