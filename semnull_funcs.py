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
import semsim_funcs

def get_semnull(pos_words, neg_words, template_sentence, template_pos):
    null_distr_scores = []
    template_words = re.split(r'[ (),;.-]', template_sentence)
    if '$' not in template_words:
        return null_distr_scores
    if len(pos_words) == 0 and len(neg_words) == 0:
        return null_distr_scores
    random_word_index = template_words.index('$')
    n_random_desired = 500
    max_iters_since_last_found = 1000
    max_iters_total = 50000
    iIter = 0
    iIterSinceLastFound = 0
    while iIterSinceLastFound < max_iters_since_last_found and iIter < max_iters_total and len(null_distr_scores) < n_random_desired:
        iIterSinceLastFound = iIterSinceLastFound + 1
        random_words = random.sample(word_vectors.index_to_key, 1)
        random_sentence = template_sentence.replace('$', random_words[0])
        random_sentence_words = random_sentence.split()
        pt_vec = pos_tag(random_sentence_words)
        pos_tag_random = pt_vec[random_word_index][1]
        if len(pos_tag_random) > len(template_pos):
            pos_tag_random = pos_tag_random[0:len(template_pos)]
        if len(template_pos) == 0 or pos_tag_random == template_pos:
            target_random = [random_words[0]]
            #print(random_words[0])
            sim_random, output_random_target_words, output_pos_words, output_neg_words, error_message = semsim_funcs.get_sims(target_random, pos_words, neg_words)
            if len(sim_random) > 0:
                null_distr_scores.append(sim_random[0])
                iIterSinceLastFound = 0
        iIter = iIter + 1
    return null_distr_scores

#d = get_semnull("owl", "The animal is $", "JJ")

#target_words="owl,tiger"; pos_words="good,lovely"; neg_words="bad,evil"
#target_words = ''.join(target_words.lower().split()).split(',')
#pos_words = ''.join(pos_words.lower().split()).split(',')
#neg_words = ''.join(neg_words.lower().split()).split(',')

def get_p(target_words, pos_words, neg_words, scores_to_test_str, null_distr_scores, contrast_word=''):
    target_words_scores, target_words, output_pos_words, output_neg_words, error_message = semsim_funcs.get_sims(target_words, pos_words, neg_words, contrast_word=contrast_word)
    if len(scores_to_test_str) > 0:
        scores_to_test = [float(s) for s in scores_to_test_str.split(',')]
    else:
        scores_to_test = []
    null_distr_scores = np.array(null_distr_scores)
    p_values = []
    target_words_scores.extend(scores_to_test)
    for score in target_words_scores:
        if len(null_distr_scores) > 0:
            this_p = sum(null_distr_scores > score) / len(null_distr_scores)
        else:
            this_p = 666
        p_values.append(this_p)
    return p_values, target_words_scores, target_words, scores_to_test, error_message

# p_values, scores_to_test_output, target_words_output = get_p('1, 5, 3, 4', [10, 1, 0, 20, 30, 44], 'owl')
