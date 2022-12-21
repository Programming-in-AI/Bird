import torch
import torch.nn as nn
import _pickle as pickle
import numpy as np
import torch.nn.functional as F


def class_embedding(sentence, word2vec, emb_dim):
    # print('Sentence shape: ', sentence.shape)
    # print('Word2vec shape: ', word2vec.shape)
    # print('Word2index shape: ', len(word2index))
    # print('Embedding dimension: ', emb_dim)
    # print(sentence)
    batch, num_class, max_words = sentence.shape # num_class is the number of top-k class ids
    # print(batch, num_class, max_words)
    rnn_size = 1024
    sentence = sentence.reshape(batch * num_class, max_words)
    sentence = sentence.long()

    # create word embedding
    embed_ques_W = word2vec.clone().detach().requires_grad_(True)
    embed_ques_W = torch.nn.Parameter(embed_ques_W)

    # create LSTM
    lstm_1 = nn.LSTM(emb_dim, rnn_size, batch_first=True)
    lstm_dropout_1 = nn.Dropout(0.2, inplace=True)
    lstm_2 = nn.LSTM(rnn_size, rnn_size, batch_first=True)
    lstm_dropout_2 = nn.Dropout(0.2, inplace=True)

    state = (torch.zeros(1, num_class, rnn_size),
             torch.zeros(1, num_class, rnn_size))

    for i in range(max_words):
        # print(sentence)
        # print(sentence.size())
        # print(i, sentence[:, i])
        # print(embed_ques_W)
        # print(embed_ques_W.size())
        cls_emb_linear = F.embedding(sentence[:, i], embed_ques_W)
        cls_emb_drop = F.dropout(cls_emb_linear, .8)
        cls_emb = torch.tanh(cls_emb_drop)
        cls_emb = cls_emb.view(batch, num_class, emb_dim)
        cls_emb = cls_emb.permute(1, 0, 2)

        # print(cls_emb.shape, state[0].shape, state[1].shape)
        output, state = lstm_1(lstm_dropout_1(cls_emb), state)
        output, state = lstm_2(lstm_dropout_2(output), state)

    output = output.reshape(batch, rnn_size, num_class)
    return output


def sentences_to_indices(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()`

    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this.

    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """

    m = X.shape[0]  # number of training examples
    # Initialize X_indices as a numpy matrix of zeros and the correct shape (≈ 1 line)
    X_indices = np.zeros((m, max_len))

    for i in range(m):  # loop over training examples
        # Convert the ith training sentence in lower case and split is into words. You should get a list of words.
        sentence_words = (X[i].lower()).split()
        sentence_words = sentence_words[:max_len]
        # Initialize j to 0
        j = 0
        # Loop over the words of sentence_words
        for w in sentence_words:
            # Set the (i,j)th entry of X_indices to the index of the correct word.
            X_indices[i, j] = word_to_index[w]
            # Increment j to j + 1
            #j = j + 1
    return X_indices


def read_glove_vecs(glove_file, dictionary_file):
    d = pickle.load(open(dictionary_file, 'rb'))
    word_to_vec_map = np.load(glove_file)
    words_to_index = d[0]
    index_to_words = d[1]
    return words_to_index, index_to_words, word_to_vec_map
