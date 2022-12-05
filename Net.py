import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 맨처음에 3개가 들어오잖아 RGB, 32개필터를 만들어버려, 필터사이즈는 3이야
        # self.conv1 = nn.Conv2d(input, output, filtersize, stride=2, padding=1)
        # self.conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        # self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        # self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        #
        # self.fc1 = nn.Linear(64 * 4 * 4, 512)
        # self.fc2 = nn.Linear(512, 200)
        #
        # self.relu = nn.ReLU()
        # self.maxPool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.ups1 = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv6 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.ups2 = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv7 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(32)
        self.ups3 = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv8 = nn.Conv2d(32, 3, 3, stride=1, padding=1)

        self.th = nn.Tanh()
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(3 * 96 * 96, 200)

    # O = (Input image size - Kernel size  + 2 * Padding size)/2 + 1
    def forward(self, x):
        # x = x  # batch_size x 3 x 256 x 256
        #
        # x = self.conv1(x)  # batch_size x 16 x 128 x 128
        # x = self.relu(x)
        # x = self.maxPool(x)  # batch_size x 16 x 64 x 64
        #
        # x = self.conv2(x)  # batch_size x 32 x 32 x 32
        # x = self.relu(x)
        # x = self.maxPool(x) # batch_size x 64 x 16 x 16
        #
        # x = self.conv3(x)  # bach_size x 128 x 8 x 8
        # x = self.relu(x)
        # x = self.maxPool(x)  # batch_size x 128 x 4 x 4
        #
        #
        # x = x.view(-1, 64 * 4 * 4)  # batch_size x 128 * 4 * 4
        # x = self.fc1(x)  # batch_size x 12
        # x = self.fc2(x)
        x1 = self.conv1(x)  # batch_size x 32 x 128 x 128
        x1 = self.relu(x1)

        x2 = self.conv2(x1)  # batch_size x 64 x 64 x 64
        x2 = self.bn1(x2)
        x2 = self.relu(x2)

        x3 = self.conv3(x2)  # bach_size x 128 x 8 x 8
        x3 = self.bn2(x3)
        x3 = self.relu(x3)

        x4 = self.conv4(x3)  # batch_size x 128 x 4 x 4
        x4 = self.bn3(x4)
        x4 = self.relu(x4)

        x5 = self.conv5(x4)  # batch_size x 128 x 2 x 2
        x5 = self.bn4(x5)
        x5 = self.relu(x5)

        x5 = x5 + x3

        x6 = self.ups1(x5)
        x6 = self.conv6(x6)  # batch_size x 64 x 4 x 4
        x6 = self.bn5(x6)
        x6 = self.relu(x6)

        x6 = x6 + x2

        x7 = self.ups2(x6)
        x7 = self.conv7(x7)  # batch_size x 32 x 8 x 8
        x7 = self.bn6(x7)
        x7 = self.relu(x7)

        x7 = x7 + x1

        x8 = self.ups3(x7)
        x8 = self.conv8(x8)  # batch_size x 3 x 16 x 16
        x8 = self.th(x8)

        x9 = self.flatten(x8)
        x9 = self.dropout(x9)
        x9 = self.fc(x9)
        return x
