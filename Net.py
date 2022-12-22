import torch.nn as nn
import torch.nn.functional as F
import torch

class ClassEmbedding(nn.Module):
    def __init__(self, sentence, word2vec, emb_dim):
        super(ClassEmbedding, self).__init__()
        _, self.num_class, self.max_words = sentence.shape
        self.word2vec = word2vec
        self.emb_dim = emb_dim

        # num_class is the number of top-k class ids
        self.rnn_size = 1024

        # create word embedding
        self.embed_ques_W = self.word2vec.clone().detach().requires_grad_(True)
        self.embed_ques_W = nn.Parameter(self.embed_ques_W)

        # create LSTM
        self.lstm_1 = nn.LSTM(self.emb_dim, self.rnn_size, batch_first=True)
        self.lstm_dropout_1 = nn.Dropout(0.2, inplace=False)
        self.lstm_2 = nn.LSTM(self.rnn_size, self.rnn_size, batch_first=True)
        self.lstm_dropout_2 = nn.Dropout(0.2, inplace=False)

        self.state = (torch.zeros(1, self.num_class, self.rnn_size),
                     torch.zeros(1, self.num_class, self.rnn_size))

    def forward(self, sentence):
        batch, _, _ = sentence.shape
        sentence = sentence.reshape(batch * self.num_class, self.max_words)
        sentence = sentence.long()
        for i in range(self.max_words):
            # print(self.sentence)
            # print(self.sentence.size())
            # print(i, self.sentence[:, i])
            # print(self.embed_ques_W)
            # print(self.embed_ques_W.size())
            # print(self.sentence[:, i])

            cls_emb_linear = F.embedding(sentence[:, i], self.embed_ques_W)
            cls_emb_drop = F.dropout(cls_emb_linear, .8)
            cls_emb = torch.tanh(cls_emb_drop)
            cls_emb = cls_emb.view(batch, self.num_class, self.emb_dim)
            cls_emb = cls_emb.permute(1, 0, 2)
            output, state = self.lstm_1(self.lstm_dropout_1(cls_emb), self.state)
            output, state = self.lstm_2(self.lstm_dropout_2(output), state)
        output = output.reshape(batch, self.rnn_size, self.num_class)

        return output
class FC(nn.Module):
    def __init__(self, x, out, dropout_rate, device):
        super(FC, self).__init__()
        self.out = out
        self.bs, self.ch, self.topk = x.shape
        self.dropout_rate = dropout_rate
        self.device = device

        self.fc = nn.Linear(self.ch, self.out).to(self.device)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        bs, ch, pixel = x.size()
        x = x.view(-1, ch)
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x.view(bs, self.out, pixel)
        return x


class FC2(nn.Module):
    def __init__(self, out, device):
        super(FC2, self).__init__()
        self.out = out
        self.device = device

        self.fc = nn.Linear(1024, self.out).to(self.device)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        bs, in_dim = x.size()
        x = self.fc(x)
        x = self.relu(x)

        return x


class FineGrainedClassifier(nn.Module):
    def __init__(self, x, device):
        super(FineGrainedClassifier, self).__init__()
        self.device = device
        self.classes_num = 200

        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(output_size=1).to(self.device)
        self.dropout = nn.Dropout(p=0.2)
        self.conv2d = nn.Conv2d(in_channels=x.size(1), out_channels=self.classes_num, kernel_size=1, bias=False).to(self.device)

    def forward(self, x):
        x = self.adaptive_avg_pool(x)  # x= (bs, ch, H, W)
        x = self.dropout(x)  # x= (bs, ch, 1, 1)
        x = self.conv2d(x)  # x= (bs, 200, 1, 1)
        x = x.squeeze()  # x= (bs, 200)
        if x.size(0) == 200:
            x = x.unsqueeze(0)
        return x

#
# class TotalNet(nn.Module):
#     def __init__(self, ):
#         super(TotalNet, self).__init__()
#         self.out = out
#         self.bs, self.ch, self.topk = x.shape
#         self.dropout_rate = dropout_rate
#         self.device = device
#
#         self.fc = nn.Linear(self.ch, self.out).to(self.device)
#         self.relu = nn.ReLU(inplace=False)
#         self.dropout = nn.Dropout(self.dropout_rate)
#
#     def forward(self, x):
#         bs, ch, pixel = x.size()
#         x = x.view(-1, ch)
#         x = self.fc(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = x.view(bs, self.out, pixel)
#         return x