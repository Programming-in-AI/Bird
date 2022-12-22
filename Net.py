import torch.nn as nn
import torch.nn.functional as F
import torch
from ClassnameProcessing import read_glove_vecs, sentences_to_indices
import numpy as np
import _pickle as pickle


class TotalNet(nn.Module):
    def __init__(self, model, FM_model, top_k, alpha, device):
        super(TotalNet, self).__init__()
        self.logit1 = None
        self.logit2 = None
        self.model = model
        self.FM_model = FM_model
        
        self.FC1 = FC(2048, 1024, dropout_rate=.5)
        self.FC2 = FC(1024, 1024, dropout_rate=.8)
        self.FC3 = FC2(1024, 2048)
        self.FGC = FineGrainedClassifier(2048, cls_num=200)

        self.att_w = None
        self.top_k = top_k
        self.alpha = alpha
        self.device = device
        self.indices, self.word2index, self.index2word, self.word2vec = make_indices()  # indices = [200,4]

        add_emb = np.expand_dims(self.word2vec[0], 0)
        self.word2vec = np.append(self.word2vec, add_emb, axis=0)
        self.word2vec[0] = np.zeros((self.word2vec.shape[1]))
        self.word2vec = torch.Tensor(self.word2vec)
        self.CE = ClassEmbedding(self.word2vec, emb_dim=300, top_k=self.top_k, device= self.device)

    def forward(self, img):
        #########################
        # Coarse classification #
        #########################
        self.logit1 = self.model(img)
        batch_size = img.size(0)
        _, y_pred_c = torch.topk(self.logit1, k=self.top_k, dim=1)  # y_pred_c : [batch_size, k ] 하나의 이미지마다 k개의 class가 있음

        tpk_cls = select_topk_cls(self.indices, y_pred_c)  # tpk_cls =[bs, top_k, 4]
        tpk_cls = tpk_cls.to(self.device)

        ###################
        # Joint Embedding #
        ###################
        # cls_emb
        cls_emb = self.CE(tpk_cls)  # cls_emb = (bs, 1024, k)

        # ftm
        ftm = self.FM_model(img)  # ftm = [bs, 2048, 14, 14]
        bs, ch, W, H = ftm.size()
        ftm = ftm.view(bs, ch, -1)  # ftm = (bs, 2048, 196)

        # v, cls_emb, att_w
        v = self.FC1(ftm)  # v = (bs, 1024, 196)
        cls_emb = self.FC2(cls_emb)  # cls_emb = (bs, 1024, k)
        self.att_w = torch.einsum('bdv,bdq->bvq', v, cls_emb)  # att_w = (bs, 196, k)
        self.att_w = nn.Softmax(dim=1)(self.att_w)  # att_w = (bs, 196, k)

        J_emb = torch.einsum('bdv,bvq,bdq->bd', v, self.att_w, cls_emb)  # J_emb = (bs, 1024)
        J_emb = self.FC3(J_emb)  # J_emb = (bs, 2048)
        J_emb = J_emb.unsqueeze(2)  # J_emb = (bs, 2048, 1)
        J_emb = J_emb + ftm  # (bs, 2048, 1) + (bs, 2048, 196)
        # self.att_w = att_w.sum(dim=2, keepdim=True)  # att_w = （bs, 196, 1）

        J_emb = J_emb.view(bs, ch, H, W)

        ###############################
        # Fine-Grained classification #
        ###############################
        self.logit2 = self.FGC(J_emb)

        logit_mixed = self.alpha * self.logit1 + (1 - self.alpha) * self.logit2

        return logit_mixed



class ClassEmbedding(nn.Module):
    def __init__(self,  word2vec, emb_dim, top_k, device):
        super(ClassEmbedding, self).__init__()
        self.device = device
        self.top_k = top_k
        self.max_words = 4
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



    def forward(self, sentence):
        batch, num_class, _ = sentence.shape

        self.state = (torch.zeros(1, self.top_k, self.rnn_size, requires_grad=False).to(self.device),
                      torch.zeros(1, self.top_k, self.rnn_size, requires_grad=False).to(self.device))

        sentence = sentence.reshape(batch * num_class, self.max_words)
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
            cls_emb = cls_emb.view(batch, num_class, self.emb_dim)
            cls_emb = cls_emb.permute(1, 0, 2)
            with torch.no_grad():
                output, self.state = self.lstm_1(self.lstm_dropout_1(cls_emb), self.state)
                output, self.state = self.lstm_2(self.lstm_dropout_2(output), self.state)

        output = output.view(batch, self.rnn_size, num_class)

        return output


class FC(nn.Module):
    def __init__(self, input_dim, out, dropout_rate):
        super(FC, self).__init__()
        self.out = out
        self.input_dim = input_dim
        self.dropout_rate = dropout_rate
        self.fc = nn.Linear(self.input_dim, self.out)
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
    def __init__(self, input_dim, out):
        super(FC2, self).__init__()
        self.out = out
        self.input_dim = input_dim
        self.fc = nn.Linear(self.input_dim, self.out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)

        return x


class FineGrainedClassifier(nn.Module):
    def __init__(self, input_dim, cls_num):
        super(FineGrainedClassifier, self).__init__()
        self.cls_num = cls_num
        self.input_dim = input_dim
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.dropout = nn.Dropout(p=0.2)
        self.conv2d = nn.Conv2d(in_channels=self.input_dim, out_channels=self.cls_num, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.adaptive_avg_pool(x)  # x= (bs, ch, H, W)
        x = self.dropout(x)  # x= (bs, ch, 1, 1)
        x = self.conv2d(x)  # x= (bs, 200, 1, 1)
        x = x.squeeze()  # x= (bs, 200)
        if x.size(0) == 200:
            x = x.unsqueeze(0)
        return x

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