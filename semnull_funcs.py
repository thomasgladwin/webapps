import os
import re
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
# word_vectors = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True, limit=200000)
# word_vectors.save('wvsubset')
wv_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wvsubset")
word_vectors = KeyedVectors.load(wv_filepath, mmap='r')
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r"\w+")
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import nltk
#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger')
import random

def get_semnull(target_word, contrast_word, template_sentence, template_pos):
    null_distr_scores = []
    template_words = re.split(r'[ (),;.-]', template_sentence)
    if '$' not in template_words or target_word not in word_vectors.key_to_index:
        return null_distr_scores
    if len(contrast_word) > 0 and contrast_word not in word_vectors.key_to_index:
        return null_distr_scores
    random_word_index = template_words.index('$')
    n_random_desired = 500
    max_iters_total = 50000
    iIter = 0
    while iIter < max_iters_total and len(null_distr_scores) < n_random_desired:
        random_words = random.sample(word_vectors.index_to_key, 1)
        random_sentence = template_sentence.replace('$', random_words[0])
        random_sentence_words = random_sentence.split()
        pt_vec = pos_tag(random_sentence_words)
        pos_tag_random = pt_vec[random_word_index][1]
        if len(template_pos) == 0 or pos_tag_random == template_pos:
            #print(random_words[0])
            sim_random = np.dot(word_vectors[target_word], word_vectors[random_words[0]])
            if len(contrast_word) > 0:
                sim_contrast = np.dot(word_vectors[contrast_word], word_vectors[random_words[0]])
                sim_random = sim_random - sim_contrast
            null_distr_scores.append(sim_random)
        iIter = iIter + 1
    return null_distr_scores

#d = get_semnull("owl", "The animal is $", "JJ")

def get_p(scores_to_test_str, null_distr_scores):
    scores_to_test = [float(s) for s in scores_to_test_str.split(',')]
    null_distr_scores = np.array(null_distr_scores)
    p_values = []
    for score in scores_to_test:
        if len(null_distr_scores) > 0:
            this_p = sum(null_distr_scores > score) / len(null_distr_scores)
        else:
            this_p = 666
        p_values.append(this_p)
    return p_values, scores_to_test

# p_values, scores_to_test_output = get_p('1, 5, 3, 4', [10, 1, 0, 20, 30, 44])