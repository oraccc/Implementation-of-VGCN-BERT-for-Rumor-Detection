from math import log
import os
import sys
import numpy as np
import scipy.sparse as sp
import pickle as pkl
import re
import time
import argparse
import nltk
import pandas as pd

from sklearn.utils import shuffle
from turtle import window_width
from nltk.corpus import stopwords
from tqdm import tqdm
from get_data import DataReader
from utils import *
from pytorch_pretrained_bert import BertTokenizer

# STEP 1: CONFIGS FOR DATA PREPARE

parser = argparse.ArgumentParser()
# ds = dataset, sw = stopwords
parser.add_argument('--ds', type=str, default='pheme')
parser.add_argument('--sw', type=int, default=1)
args = parser.parse_args()

config_dataset = args.ds
config_use_stopwords = True if args.sw else False


dump_dir = './prepared_data'
if not os.path.exists(dump_dir):
    os.mkdir(dump_dir)

if config_use_stopwords:
    freq_min_for_word_choice = 10
else:
    freq_min_for_word_choice = 1


window_size = 1000  # word co-occurence with context windows, use whole sentence

tfidf_mode = 'all_tfidf'  # only_tf / all_tfidf


use_tokenizer_at_clean_string = True

bert_model_scale = 'bert-base-uncased' # bert-base-uncased / bert-large-uncased
bert_lower_case = True

print('\n')
print('-----------STEP 1: CONFIGS FOR DATA PREPARE-----------')
print('Dataset: ', config_dataset)
print('Min Frequency for Word Choice: ', freq_min_for_word_choice)
print('Window Size: ', window_size)
print('Will Delete Stop Words: ', config_use_stopwords,)
print('Will Use Bert Tokenizer at Clean: ', use_tokenizer_at_clean_string)
print('TF-IDF Mode: ', tfidf_mode)
print('Bert Model Scale: ', bert_model_scale)
print('Bert Lower Case: ', bert_lower_case)



# STEP 2.1: GET TWEETS, LABELS, CONFIDENCE FROM DATA FILE

print('\n')
print('-----------STEP 2: GET TWEETS, LABELS, CONFIDENCE FROM DATA FILE-----------')

start = time.time()

train, test = DataReader(
    "data/PHEME-SEG/train.txt", "./data/PHEME-SEG/test.txt").read()

train_size = len(train)
test_size = len(test)
print('PHEME Dataset, train_szie: %d, test_size: %d' %
      (train_size, test_size))

trainset = {}
testset = {}

for data, dataset in [(train, trainset), (test, testset)]:
    label = []
    all_text = []
    for line in data:
        label.append(line[0])
        all_text.append(line[1])
    dataset["label"] = label
    dataset["data"] = all_text

label_to_index = {label: i for i, label in enumerate(testset['label'])}
index_to_label = {i: label for i, label in enumerate(testset['label'])}

corpus = trainset['data'] + testset['data']
label = np.array(trainset['label'] + testset['label'])
corpus_size = len(corpus)
label_prob = np.eye(corpus_size, len(label_to_index))[label]

doc_content_list = []
for t in corpus:
    doc_content_list.append(del_http_user_tokenize(t))

# STEP 2.2: GET STATISTICS FOR ORIGINAL TEXT DATA

max_len_seq = 0
max_len_seq_idx = -1
min_len_seq = 1000
min_len_seq_idx = -1
sen_len_list = []

for i, seq in enumerate(doc_content_list):
    seq = seq.split()
    sen_len_list.append(len(seq))
    if len(seq) < min_len_seq:
        min_len_seq = len(seq)
        min_len_seq_idx = i
    if len(seq) > max_len_seq:
        max_len_seq = len(seq)
        max_len_seq_idx = i

print('Statistics for original text')
print('max_len: %d, max_len_id: %d, min_len: %d, min_len_id: %d, avg_len: %.2f'
      % (max_len_seq, max_len_seq_idx, min_len_seq, min_len_seq_idx, np.array(sen_len_list).mean()))


# STEP 3.1: REMOVE STOP WORDS FROM TWEETS

print('\n')
print('-----------STEP 3: TOKENIZE SENTENCES & REMOVE STOP WORDS FROM TEXTS-----------')

if config_use_stopwords:
    from nltk.corpus import stopwords
    nltk.download('stopwords')
    stop_words = stopwords.words('english')
    stop_words = set(stop_words)
else:
    stop_words = {}
print('Stop words:', stop_words)


tmp_word_freq = {}  # to remove rare words
new_doc_content_list = []

# use bert_tokenizer for split the sentence
if use_tokenizer_at_clean_string:
    print('Use bert_tokenizer for seperate words to bert vocab')
    bert_tokenizer = BertTokenizer.from_pretrained(
        bert_model_scale, do_lower_case=bert_lower_case)

for doc_content in doc_content_list:
    new_doc = clean_str(doc_content)

    if use_tokenizer_at_clean_string:
        sub_words = bert_tokenizer.tokenize(new_doc)
        sub_doc = ' '.join(sub_words).strip()
        new_doc = sub_doc
    new_doc_content_list.append(new_doc)

    for word in new_doc.split():
        if word in tmp_word_freq:
            tmp_word_freq[word] += 1
        else:
            tmp_word_freq[word] = 1

doc_content_list = new_doc_content_list


clean_docs = []
count_void_doc = 0

for i, doc_content in enumerate(doc_content_list):
    words = doc_content.split()
    doc_words = []

    for word in words:
        if word not in stop_words and tmp_word_freq[word] >= freq_min_for_word_choice:
            doc_words.append(word)

    doc_str = ' '.join(doc_words).strip()

    if doc_str == '':
        count_void_doc += 1
        print('No.', i, 'is a empty doc after treat, replaced by \'%s\'. original: %s' % (
            doc_str, doc_content))
    clean_docs.append(doc_str)


print('Total', count_void_doc, ' docs are empty.')


# STEP 3.2: GET STATISTICS FOR SPLIT AND CLEANED TEXT DATA

min_len = 10000
min_len_id = -1
max_len = 0
max_len_id = -1
aver_len = 0

for i, line in enumerate(clean_docs):
    temp = line.strip().split()
    aver_len = aver_len + len(temp)
    if len(temp) < min_len:
        min_len = len(temp)
        min_len_id = i
    if len(temp) > max_len:
        max_len = len(temp)
        max_len_id = i

aver_len = 1.0 * aver_len / len(clean_docs)
print('Statistics after stopwords and tokenizer:')
print('min_len : ' + str(min_len) + ' min_len_id: ' + str(min_len_id))
print('max_len : ' + str(max_len) + ' max_len_id: ' + str(max_len_id))
print('average_len : ' + str(aver_len))


# STEP 4.1: PREPARE DATA FOR BUILD GRAPH

# train_docs = clean_docs[: train_size]
# test_docs = clean_docs[train_size : train_size + test_size]

train_label = label[: train_size]
test_label = label[train_size: train_size + test_size]

train_label_prob = label_prob[: train_size]
test_label_prob = label_prob[train_size: train_size + test_size]

# build vocab using whole corpus(train + test + genelization)
word_set = set()
for doc_words in clean_docs:
    words = doc_words.split()
    for word in words:
        word_set.add(word)

vocab = list(word_set)
vocab_size = len(vocab)

vocab_map = {}
for i in range(vocab_size):
    vocab_map[vocab[i]] = i

word_doc_list = {}
for i in range(len(clean_docs)):
    doc_words = clean_docs[i]
    words = doc_words.split()
    appeared = set()
    for word in words:
        if word in appeared:
            continue
        if word in word_doc_list:
            doc_list = word_doc_list[word]
            doc_list.append(i)
            word_doc_list[word] = doc_list
        else:
            word_doc_list[word] = [i]
        appeared.add(word)

word_doc_freq = {}
for word, doc_list in word_doc_list.items():
    word_doc_freq[word] = len(doc_list)


# STEP 4.2: BUILD DOCUMENT WORD HETEROGENEOUS GRAPH AND VOCABULARY GRAPH

print('\n')
print('----------STEP 4: BUILD GRAPH----------')

print('Calculate first isomerous adj and first isomorphic vocab adj, get word-word PMI values')

adj_label = np.hstack((train_label, np.zeros(vocab_size), test_label))
adj_label_prob = np.vstack((train_label_prob, np.zeros((vocab_size, len(
    label_to_index)), dtype=np.float32), test_label_prob))

windows = []
for doc_words in clean_docs:
    words = doc_words.split()
    length = len(words)
    if length <= window_size:
        windows.append(words)
    else:
        for j in range(length - window_size + 1):
            window = words[j: j + window_size]
            windows.append(window)

print('Train_valid size:', len(clean_docs), 'Window number:', len(windows))

word_window_freq = {}
for window in windows:
    appeared = set()
    for i in range(len(window)):
        if window[i] in appeared:
            continue
        if window[i] in word_window_freq:
            word_window_freq[window[i]] += 1
        else:
            word_window_freq[window[i]] = 1
        appeared.add(window[i])



word_pair_count = {}

for window in tqdm(windows):
    appeared = set()

    for i in range(1, len(window)):
        for j in range(0, i):
            word_i = window[i]
            word_i_id = vocab_map[word_i]
            word_j = window[j]
            word_j_id = vocab_map[word_j]

            if word_i_id == word_j_id:
                continue
            word_pair_str = str(word_i_id) + ',' + str(word_j_id)
            if word_pair_str in appeared:
                continue
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1
            appeared.add(word_pair_str)
            # two orders
            word_pair_str = str(word_j_id) + ',' + str(word_i_id)
            if word_pair_str in appeared:
                continue
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1
            appeared.add(word_pair_str)


row = []
col = []
weight = []
tfidf_row = []
tfidf_col = []
tfidf_weight = []
vocab_adj_row = []
vocab_adj_col = []
vocab_adj_weight = []

num_window = len(windows)
tmp_max_npmi = 0
tmp_min_npmi = 0
tmp_max_pmi = 0
tmp_min_pmi = 0

for key in word_pair_count:
    temp = key.split(',')
    i = int(temp[0])
    j = int(temp[1])
    count = word_pair_count[key]
    word_freq_i = word_window_freq[vocab[i]]
    word_freq_j = word_window_freq[vocab[j]]

    pmi = log((1.0 * count / num_window) /
              (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))

    npmi = log(1.0 * word_freq_i * word_freq_j / (num_window *
               num_window)) / log(1.0 * count / num_window) - 1

    if npmi > tmp_max_npmi:
        tmp_max_npmi = npmi
    if npmi < tmp_min_npmi:
        tmp_min_npmi = npmi
    if pmi > tmp_max_pmi:
        tmp_max_pmi = pmi
    if pmi < tmp_min_pmi:
        tmp_min_pmi = pmi
    if pmi > 0:
        row.append(train_size + i)
        col.append(train_size + j)
        weight.append(pmi)
    if npmi > 0:
        vocab_adj_row.append(i)
        vocab_adj_col.append(j)
        vocab_adj_weight.append(npmi)

print('max_pmi:', tmp_max_pmi, 'min_pmi:', tmp_min_pmi)
print('max_npmi:', tmp_max_npmi, 'min_npmi:', tmp_min_npmi)


# STEP 5.1: CALCULATE DOC-WORD TF-IDF WEIGHT

print('Calculate doc-word tf-idf weight')

n_docs = len(clean_docs)
doc_word_freq = {}
for doc_id in range(n_docs):
    doc_words = clean_docs[doc_id]
    words = doc_words.split()
    for word in words:
        word_id = vocab_map[word]
        doc_word_str = str(doc_id) + ',' + str(word_id)
        if doc_word_str in doc_word_freq:
            doc_word_freq[doc_word_str] += 1
        else:
            doc_word_freq[doc_word_str] = 1

for i in range(n_docs):
    doc_words = clean_docs[i]
    words = doc_words.split()
    doc_word_set = set()
    tfidf_vec = []
    for word in words:
        if word in doc_word_set:
            continue
        j = vocab_map[word]
        key = str(i) + ',' + str(j)
        tf = doc_word_freq[key]
        tfidf_row.append(i)
        if i < train_size:
            row.append(i)
        else:
            row.append(i + vocab_size)
        tfidf_col.append(j)
        col.append(train_size + j)
        # smooth
        idf = log((1.0 + n_docs) / (1.0+word_doc_freq[vocab[j]])) + 1.0
        # weight.append(tf * idf)
        if tfidf_mode == 'only_tf':
            tfidf_vec.append(tf)
        else:
            tfidf_vec.append(tf * idf)
        doc_word_set.add(word)
    if len(tfidf_vec) > 0:
        weight.extend(tfidf_vec)
        tfidf_weight.extend(tfidf_vec)


print('\n')
print('----------STEP 5: ASSEMBLE ADJACENCY MATRIX AND DUMP TO FILES----------')

# STEP 5.2: ASSEMBLE ADJACENCY MATRIX AND DUMP TO FILES

node_size = vocab_size + corpus_size

adj_list = []
adj_list.append(sp.csr_matrix((weight, (row, col)),
                shape=(node_size, node_size), dtype=np.float32))
for i, adj in enumerate(adj_list):
    adj_list[i] = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_list[i].setdiag(1.0)

vocab_adj = sp.csr_matrix((vocab_adj_weight, (vocab_adj_row, vocab_adj_col)), shape=(
    vocab_size, vocab_size), dtype=np.float32)
vocab_adj.setdiag(1.0)

print('Calculate isomorphic vocab adjacency matrix using doc\'s tf-idf...')
tfidf_all = sp.csr_matrix((tfidf_weight, (tfidf_row, tfidf_col)), shape=(
    corpus_size, vocab_size), dtype=np.float32)
tfidf_train = tfidf_all[:train_size]
tfidf_test = tfidf_all[train_size:train_size + test_size]

tfidf_X_list = [tfidf_train, tfidf_test]
vocab_tfidf = tfidf_all.T.tolil()
for i in range(vocab_size):
    norm = np.linalg.norm(vocab_tfidf.data[i])
    if norm > 0:
        vocab_tfidf.data[i] /= norm
vocab_adj_tf = vocab_tfidf.dot(vocab_tfidf.T)

# check
print('Check adjacent matrix...')
for k in range(len(adj_list)):
    count = 0
    for i in range(adj_list[k].shape[0]):
        if adj_list[k][i, i] <= 0:
            count += 1
            print('No.%d adj, abnormal diagonal found, No.%d' % (k, i))
    if count > 0:
        print('No.%d adj, total %d zero diagonal found.' % (k, count))

# dump objects

print('Dump objects...')

with open(dump_dir+"/data_%s.index_label" % config_dataset, 'wb') as f:
    pkl.dump([label_to_index, index_to_label], f)

with open(dump_dir+"/data_%s.vocab_map" % config_dataset, 'wb') as f:
    pkl.dump(vocab_map, f)

with open(dump_dir+"/data_%s.vocab" % config_dataset, 'wb') as f:
    pkl.dump(vocab, f)

with open(dump_dir+"/data_%s.adj_list" % config_dataset, 'wb') as f:
    pkl.dump(adj_list, f)
with open(dump_dir+"/data_%s.label" % config_dataset, 'wb') as f:
    pkl.dump(label, f)
with open(dump_dir+"/data_%s.label_prob" % config_dataset, 'wb') as f:
    pkl.dump(label_prob, f)
with open(dump_dir+"/data_%s.train_label" % config_dataset, 'wb') as f:
    pkl.dump(train_label, f)
with open(dump_dir+"/data_%s.train_label_prob" % config_dataset, 'wb') as f:
    pkl.dump(train_label_prob, f)
with open(dump_dir+"/data_%s.test_label" % config_dataset, 'wb') as f:
    pkl.dump(test_label, f)
with open(dump_dir+"/data_%s.test_label_prob" % config_dataset, 'wb') as f:
    pkl.dump(test_label_prob, f)
with open(dump_dir+"/data_%s.tfidf_list" % config_dataset, 'wb') as f:
    pkl.dump(tfidf_X_list, f)
with open(dump_dir+"/data_%s.vocab_adj_pmi" % (config_dataset), 'wb') as f:
    pkl.dump(vocab_adj, f)
with open(dump_dir+"/data_%s.vocab_adj_tf" % (config_dataset), 'wb') as f:
    pkl.dump(vocab_adj_tf, f)
with open(dump_dir+"/data_%s.clean_docs" % config_dataset, 'wb') as f:
    pkl.dump(clean_docs, f)

print('Data prepared, spend %.2f s' % (time.time()-start))
