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
#python -m spacy download en_core_web_sm
import spacy
nlp = spacy.load('en_core_web_sm')
import numpy as np

def carry_forward_backward(subject_list):
    # Forward
    current_subj = ''
    for n in range(len(subject_list)):
        if subject_list[n] == '':
            subject_list[n] = current_subj
        else:
            current_subj = subject_list[n]
    # Backward
    current_subj = ''
    for n in range(len(subject_list)):
        if subject_list[-1 - n] == '':
            subject_list[-1 - n] = current_subj
        else:
            current_subj = subject_list[-1 - n]
    return subject_list

def lists_to_dict(subject_list, attr_list):
    Knowledge = {}
    for n in range(len(subject_list)):
        subject = subject_list[n]
        if subject not in Knowledge.keys():
            Knowledge.update({subject: attr_list[n]})
        else:
            Knowledge[subject] = Knowledge[subject] + attr_list[n]
    return Knowledge

def get_knowledge_from_sentence(sentence):
    sentence = sentence.strip()
    doc=nlp(sentence)
    conjunctions_list = [tok.text for tok in doc if (tok.dep_ in ["cc", "punct"])]
    if len(conjunctions_list) > 0:
        sub_sentences = sentence.split(conjunctions_list[0])
        subject_list = []
        attr_list = []
        for sub_sentence in sub_sentences:
            sub_subject_list, sub_attr_list = get_knowledge_from_sentence(sub_sentence)
            subject_list = subject_list + sub_subject_list
            attr_list = attr_list + sub_attr_list
    else:
        subject_list = [(i_tok[0], i_tok[1].lemma_) for i_tok in enumerate(doc) if (i_tok[1].dep_ in ["nsubj", "npadvmod"] and i_tok[1].tag_ in ["NNP", "NN", "NNS"])] # Prefer nouns to pronouns
        if len(subject_list) == 0:
            subject_list = [(i_tok[0], i_tok[1].lemma_) for i_tok in enumerate(doc) if (i_tok[1].tag_ in ["NNP", "NN", "NNS"])]
            if len(subject_list) == 0:
                subject_list = [(i_tok[0], i_tok[1].lemma_) for i_tok in enumerate(doc) if (i_tok[1].dep_ in ["nsubj", "npadvmod"])]
                if len(subject_list) == 0:
                    subject_list = [(0, '')]
        attr_list = [[] for subj in subject_list]
        negation = ''
        for index_tok in enumerate(doc):
            index = index_tok[0]
            tok = index_tok[1]
            if tok.dep_ in ["neg"]:
                negation = 'not '
            if tok.dep_ in ["attr", "dobj", "acomp", "amod", "advmod", "pobj"] or (tok.pos_ in ["VERB"] and tok.lemma_ not in ['be', 'have', 'feel', 'think']) or (len(subject_list[0][1]) == 0 and tok.dep_ in ['ROOT']):
                distances = [np.abs(s[0] - index) for s in subject_list]
                link_to = np.argmin(distances)
                attr_list[link_to].append(negation + tok.lemma_)
                object_of_verb = ''
                if tok.pos_ in ["VERB"] and len([tmptok.dep_ for tmptok in doc if (tmptok.dep_ == "dobj")]) > 0:
                    dobjs = [tmptok.lemma_ for tmptok in doc if (tmptok.dep_ == "dobj")]
                    object_of_verb = ' ' + dobjs[0]
                attr_list[link_to].append(negation + tok.lemma_ + object_of_verb)
        subject_list = [s[1] for s in subject_list]
    subject_list = carry_forward_backward(subject_list)
    attr_list = [list(set(a)) for a in attr_list]
    return subject_list, attr_list

def add_to_knowledge(Knowledge, Knowledge_sentence, par_index=0):
    for subject in Knowledge_sentence.keys():
        if subject not in Knowledge.keys():
            Knowledge.update({subject: {attr: {'count':1, 'par_index':[par_index]} for attr in Knowledge_sentence[subject]}})
        else:
            for attr in Knowledge_sentence[subject]:
                if attr in Knowledge[subject].keys():
                    Knowledge[subject][attr]['count'] = Knowledge[subject][attr]['count'] + 1
                    Knowledge[subject][attr]['par_index'].append(par_index)
                else:
                    Knowledge[subject].update({attr: {'count':1, 'par_index': [par_index]}})
    return Knowledge

def Knowledge_to_list(Knowledge):
    knowledge_list = []
    # Order subjects by highest count over all attributes.
    max_count_per_subject = [(k[0], np.max([c[1]['count'] for c in k[1].items()])) for k in Knowledge.items()]
    counts = [c[1] for c in max_count_per_subject]
    ind_order = list(np.argsort(counts))
    ind_order.reverse()
    for n in ind_order:
        this_subject = max_count_per_subject[n][0]
        this_list = [this_subject]
        this_list.append([[a[0], a[1]['count'], a[1]['par_index']] for a in Knowledge[this_subject].items()])
        knowledge_list.append(this_list)
    return knowledge_list

def order_attr_lists(knowledge_list):
    for k in knowledge_list:
        attr_list = k[1]
        counts = [e[1] for e in attr_list]
        indices = np.flip(np.argsort(counts))
        attr_list_sorted = [attr_list[n] for n in indices]
        k[1] = attr_list_sorted
    return knowledge_list

def gather_subjects_from_attr(knowledge_list):
    for ik1 in range(len(knowledge_list)):
        k1 = knowledge_list[ik1]
        for ik2 in range(ik1+1, len(knowledge_list)):
            k2 = knowledge_list[ik2]
            index_attr_is_subj = np.argwhere([a[0] == k1[0] for a in k2[1]])
            if len(index_attr_is_subj) > 0:
                index_attr_is_subj = index_attr_is_subj[0][0]
                # Add k2's subject as an attribute of k1
                existing_index = np.argwhere([a[0] == k2[0] for a in k1[1]])
                if len(existing_index) == 0:
                    k1[1].append([k2[0], k2[1][index_attr_is_subj][1], k2[1][index_attr_is_subj][2]])
                else:
                    existing_index = existing_index[0][0]
                    # Check whether a new paragraph; if not, increment counter.
                    to_add = np.argwhere([par_index not in k1[1][existing_index][2] for par_index in k2[1][index_attr_is_subj][2]])
                    if len(to_add) > 0:
                        to_add = to_add[0]
                        for n in to_add:
                            k1[1][existing_index][1] = k1[1][existing_index][1] + 1
                            k1[1][existing_index][2].append(k2[1][index_attr_is_subj][2][n])
    return knowledge_list

def get_textCoder(text):
    Knowledge = {} # subject:attr:count
    paragraphs = text.split('\n')
    par_index = 0
    for paragraph in paragraphs:
        if len(paragraph) == 0:
            continue
        Knowledge_paragraph = {}
        sentences = [paragraph] # paragraph.split('.') # Use full paragraph to fill missings.
        for sentence in sentences:
            if len(sentence) == 0:
                continue
            subject_list, attr_list = get_knowledge_from_sentence(sentence)
            Knowledge_sentence = lists_to_dict(subject_list, attr_list)
            Knowledge_paragraph = add_to_knowledge(Knowledge_paragraph, Knowledge_sentence)
        Knowledge = add_to_knowledge(Knowledge, Knowledge_paragraph, par_index)
        par_index = par_index + 1
    knowledge_list = Knowledge_to_list(Knowledge)
    knowledge_list = order_attr_lists(knowledge_list)
    knowledge_list = gather_subjects_from_attr(knowledge_list)
    return knowledge_list

if False:
    text = 'I like Pepsi. Thomas is smelly.\nSmelly.\nPepsi is great.\nI think Pepsi is great.\nThomas is a smelly smelly smelly duck.\nJim is happy. Tim is dead. Thomas is smelly. Tim hates food.'
    text = 'Cats are evil.\nCats are evil.\nCats are evil.\nI see evil cats.The evil of cats astounds me.\nThe evil of cats astounds me.'
    knowledge_list = get_textCoder(text)
    for k in knowledge_list:
        print(k)

    sentence = "The evil of cats astounds me."
    doc=nlp(sentence)
    sub_toks = [tok for tok in doc if (tok.dep_ == "nsubj") ]
    print(sub_toks)
    for token in doc:
        print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
                token.shape_, token.is_alpha, token.is_stop)
    print(sub_toks[0].pos_)
    spacy.explain('pobj')
