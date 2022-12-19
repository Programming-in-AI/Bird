# Embed the class names of CUB-200-2011 into a vector space using word2vec

import os
import sys
import numpy as np
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import nltk

# Read the class names
class_names = []

# Path to the class names
data_dir = 'CUB_200_2011/'
class_name_file = os.path.join(data_dir, 'classes.txt')

with open(class_name_file, 'r') as f:
    for line in f:
        class_names.append(line.strip())

print(class_names)

# Tokenize the class names
class_names = [nltk.word_tokenize(class_name) for class_name in class_names]

print(class_names)

# Train the word2vec model
model = Word2Vec(class_names, size=128, window=5, min_count=1, workers=4)

# Save the model
model.save('class_name_embedding.model')

# Save the class name embeddings
class_name_embeddings = []

for class_name in class_names:
    class_name_embeddings.append(model[class_name])

class_name_embeddings = np.array(class_name_embeddings)
# np.save('class_name_embeddings.npy', class_name_embeddings)

print(class_name_embeddings.shape)
print(class_name_embeddings[0])