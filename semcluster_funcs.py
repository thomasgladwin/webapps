import numpy as np
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r"\w+")
from nltk import pos_tag
from gensim.models.keyedvectors import KeyedVectors
import os
from sklearn.cluster import KMeans
from kneed import KneeLocator

#word_vectors = KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin.gz', binary=True, limit=200000)
#word_vectors.save('wvsubset')
wv_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wvsubset")
word_vectors = KeyedVectors.load(wv_filepath, mmap='r')

def find_split(w):
    KL = KneeLocator(range(1, len(w)+1), w, curve="convex", direction="decreasing")
    ind_split = KL.elbow
    print(ind_split)
    return ind_split

def run_PCA(WVM):
    WVM = WVM - np.mean(WVM, axis=0)
    C = (WVM.T @ WVM)/(WVM.shape[0]-1)
    w, v = np.linalg.eig(C)
    w = np.array(np.real(w))
    v = np.matrix(np.real(v))
    ind_split = np.max([min(2,WVM.shape[1]), find_split(w)])
    w = w[0:ind_split]
    v = v[:, 0:ind_split]
    L = WVM @ v
    return v, L

def get_labels(v):
    dimension_labels = []
    for i_col in range(v.shape[1]):
        col = v[:, i_col]
        this_label = word_vectors.similar_by_vector(np.ravel(col))
        dimension_labels.append(this_label[0])
    return dimension_labels

def run_dimensions(WVM):
    v, L = run_PCA(WVM)
    dimension_labels = get_labels(v)
    return dimension_labels, L, v

def get_clusters_inner(L, words, WVM):
    err_v = []
    for k in range(1, L.shape[0]):
        kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(np.asarray(L))
        this_err = kmeans.inertia_
        err_v.append(this_err)
    best_k = find_split(err_v)
    kmeans = KMeans(n_clusters=best_k, random_state=0, n_init="auto").fit(np.asarray(L))
    labels = kmeans.labels_
    cluster_labels = []
    cluster_items = []
    for label in np.unique(labels):
        sel = (labels == label)
        this_set = (np.array(words)[sel])
        this_WVM = WVM[sel, :]
        mean_vec = np.mean(this_WVM, axis=0)
        cluster_label = word_vectors.similar_by_vector(np.ravel(mean_vec))[0][0]
        cluster_labels.append(cluster_label)
        cluster_items.append(this_set)
    return cluster_labels, cluster_items

def get_clusters(words):
    words = words.replace('.', '')
    words = ''.join(words.split()).split(',')
    words = [w for w in words if w in word_vectors.key_to_index]
    if len(words) == 0:
        return [], []
    WVA = np.array([word_vectors[w] for w in words])
    WVM = np.matrix(WVA)
    dimension_labels, L, v = run_dimensions(WVM)
    cluster_labels, cluster_items = get_clusters_inner(L, words, WVM)
    return cluster_labels, cluster_items

# words='dog,cat,hamster,rabbit,skyscraper,house,building,bungalow,angry,happy,sad,furious,joyful.'
#words='happy,joyful,glad,gleeful,sad,down,depressed,dejected,unhappy'
#get_clusters(words)