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
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def read_words():
    words = []
    fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'emotional_words2.txt')
    fid = open(fn, 'r')
    while True:
        line = fid.readline()
        if not line:
            break
        if len(line) > 0:
            word = tokenizer.tokenize(line)
            words.append(word[0])
    fid.close()
    return words

def get_WVM():
    words = read_words()
    words = [lemmatizer.lemmatize(word) for word in words]
    emowords = [word0 for word0 in words if word0 in word_vectors.key_to_index.keys()]
    emowords = set(emowords)
    emowords = [word0 for word0 in emowords]
    WVM = np.array([word_vectors[word0] for word0 in emowords])
    return emowords, WVM

def get_emocont_inner(text):
    emowords, WVM = get_WVM()
    sentences = re.split(r'[,;.-]', text)
    sims_all = []
    for sentence in sentences:
        tokens = tokenizer.tokenize(sentence)
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        tokens = [token0 for token0 in tokens if token0 in word_vectors.key_to_index.keys()]
        if len(tokens) == 0:
            continue
        # Test for negation-tokens (for adjectives)
        neg_v = [np.dot(word_vectors['not'] - word_vectors['very'], word_vectors[token]) for token in tokens]
        neg_v = np.array(neg_v)
        nonnegation = 1 - 2 * np.mod(len(neg_v[neg_v > 1]), 2)
        # Get nouns and adjectives after preprocessing
        pt = pos_tag(tokens)
        tokens2 = [x[0] for x in pt if x[1] in ['JJ', 'NN', 'RB', 'VB']]
        if False and len(tokens2) > 0:
            tokens = tokens2
        print(tokens)
        # Find similarities with the emotion words
        token_sims = []
        for token0 in tokens:
            sims0 = [nonnegation * np.dot(word_vectors[token0], WVMv) for WVMv in WVM]
            token_sims.append(sims0)
        sims_all.append(token_sims)
    # Get strongest emotional meanings per sentence
    nEmos_per_token = 3
    emo_indices = []
    emo_sims = []
    for sentence_level in sims_all:
        for token_level in sentence_level:
            sims_token_with_emos = np.array(token_level)
            indices = np.argsort(sims_token_with_emos)
            emo_indices.append(indices[-nEmos_per_token:])
            token_level_sims = np.array(token_level)[indices]
            emo_sims.append(token_level_sims[-nEmos_per_token:])
    # Flatten over all sentences
    emo_indices = np.array(emo_indices).flatten()
    emo_sims = np.array(emo_sims).flatten()
    # Order selected emotions by their similarity level
    indices = np.argsort(emo_sims)
    indices = emo_indices[indices]
    # Get best two emotions
    nEmos_total = np.min([3, nEmos_per_token])
    output_this = []
    iEmo = 1
    nEmo = 0
    used_indices = []
    while nEmo < nEmos_total and iEmo <= len(indices):
        this_index = indices[-iEmo]
        if not this_index in used_indices:
            output_this.append(emowords[this_index])
            used_indices.append(this_index)
            nEmo = nEmo + 1
        iEmo = iEmo + 1
    print(output_this)
    # To add: valence
    # ...
    return output_this

def get_emocont(text):
    output = []
    output_all = get_emocont_inner(text)
    paragraphs = text.split('\n')
    output_per_paragraph = []
    for paragraph in paragraphs:
        if len(paragraph) == 0:
            continue
        output_this = get_emocont_inner(paragraph)
        if len(output_this) == 0:
            output_this = ['-']
        output_per_paragraph.append(output_this[0])
    return output_all, output_per_paragraph
