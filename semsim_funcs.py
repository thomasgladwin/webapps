import numpy as np
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r"\w+")
from nltk import pos_tag
from gensim.models.keyedvectors import KeyedVectors
import os

#word_vectors = KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin.gz', binary=True, limit=200000)
#word_vectors.save('wvsubset')
wv_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wvsubset")
word_vectors = KeyedVectors.load(wv_filepath, mmap='r')

def get_sims(target_words, pos_words, neg_words):
    target_words = [w for w in target_words if w in word_vectors.key_to_index]
    pos_words = [w for w in pos_words if w in word_vectors.key_to_index]
    neg_words = [w for w in neg_words if w in word_vectors.key_to_index]
    print(target_words)
    relative_sims = []
    for target_word in target_words:
        pos_sims = []
        neg_sims = []
        for attribute_word in pos_words:
            this_sim = word_vectors.similarity(target_word, attribute_word)
            pos_sims.append(this_sim)
        for attribute_word in neg_words:
            this_sim = word_vectors.similarity(target_word, attribute_word)
            neg_sims.append(this_sim)
        if len(pos_sims) > 0:
            mean_pos = np.array(pos_sims).mean()
        else:
            mean_pos=0
        if len(neg_sims) > 0:
            mean_neg = np.array(neg_sims).mean()
        else:
            mean_neg = 0
        relative_sim = mean_pos - mean_neg
        relative_sims.append(relative_sim)
    return relative_sims, target_words, pos_words, neg_words
