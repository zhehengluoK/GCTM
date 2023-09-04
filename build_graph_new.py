import os
import sys
import itertools
import math
from collections import defaultdict
from time import time
from tqdm import tqdm
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx

import file_handling as fh
from graph_utils import print_graph_detail, sparse_mx_to_torch_sparse_tensor, PrepareGraph, setup_seed
import numpy as np
import random


def get_window(content_lst, window_size):
    """
    找出窗口
    :param content_lst: [list of str]
    :param window_size:
    :return:
    """
    word_window_freq = defaultdict(int)  # w(i)  单词在窗口单位内出现的次数
    word_pair_count = defaultdict(int)  # w(i, j)
    windows_len = 0
    for words in tqdm(content_lst, desc="Split by window"):
        windows = list()

        if isinstance(words, str):
            words = words.split()
        length = len(words)

        if length <= window_size:
            windows.append(words)
        else:
            for j in range(length - window_size + 1):
                window = words[j: j + window_size]
                windows.append(sorted(list(set(window))))

        for window in windows:
            for word in window:
                word_window_freq[word] += 1

            for word_pair in itertools.combinations(window, 2):
                word_pair_count[word_pair] += 1

        windows_len += len(windows)
    return word_window_freq, word_pair_count, windows_len


def cal_pmi(W_ij, W, word_freq_1, word_freq_2):
    p_i = word_freq_1 / W
    p_j = word_freq_2 / W
    p_i_j = W_ij / W
    pmi = math.log(p_i_j / (p_i * p_j))

    return pmi


def calculate_npmi(W_ij, W, word_freq_1, word_freq_2):
    p_i = word_freq_1 / W
    p_j = word_freq_2 / W
    p_i_j = W_ij / W
    pmi = math.log(p_i_j / (p_i * p_j))

    return - pmi / math.log(p_i_j)


def count_pmi(windows_len, word_pair_count, word_window_freq, vocab):
    word_pmi_lst = list()
    for word_pair, W_i_j in tqdm(word_pair_count.items(), desc="Calculate pmi between words"):
        # 去掉不在词表中的边
        if word_pair[0] not in vocab or word_pair[1] not in vocab:
            continue

        word_freq_1 = word_window_freq[word_pair[0]]
        word_freq_2 = word_window_freq[word_pair[1]]

        pmi = cal_pmi(W_i_j, windows_len, word_freq_1, word_freq_2)       # PMI
        # pmi = calculate_npmi(W_i_j, windows_len, word_freq_1, word_freq_2)  # NPMI

        word_pmi_lst.append([word_pair[0], word_pair[1], pmi])
    return word_pmi_lst


def get_pmi_edge(content_lst, vocab, window_size=30):
    """If negative=True, calculate negative word co-occurrence graph"""
    if isinstance(content_lst, str):
        # content_lst = list(open(content_lst, 'r'))
        # for i in range(len(content_lst)):
        #     content_lst[i] = content_lst[i].split('\t')[2]
        content_lst = fh.read_text(content_lst)

    print("pmi read file len:", len(content_lst))

    pmi_start = time()
    word_window_freq, word_pair_count, windows_len = get_window(content_lst, window_size=window_size)
    print("windows number:", windows_len)
    pmi_edge_lst = count_pmi(windows_len, word_pair_count, word_window_freq, vocab)
    print("Total number of edges between word:", len(pmi_edge_lst))
    pmi_time = time() - pmi_start
    return pmi_edge_lst, pmi_time


class BuildGraph:
    def __init__(self, dataset, prefix, window_size=20, threshold_pos=0., threshold_neg=0.):
        """ prefix: str, option: train/test """
        processed_corpus_path = f"data/{dataset}/processed"
        self.graph_path = f"data/{dataset}/graph_{window_size}_{threshold_pos}"
        if not os.path.exists(self.graph_path):
            os.makedirs(self.graph_path)

        self.vocab_lst = fh.read_json(os.path.join(processed_corpus_path, 'train.vocab.json'))
        self.word2id = {word: idx for idx, word in enumerate(self.vocab_lst)}  # 单词映射
        self.dataset = dataset

        print(f"\n==> 现在的数据集是:{dataset}/{prefix} <==")

        self.pos_word_graph = nx.Graph()
        self.neg_word_graph = nx.Graph()
        # 对角线元素都设为 1，预先处理对角线元素，不然阈值调大之后，节点数量可能不够
        for i in range(len(self.vocab_lst)):
            self.pos_word_graph.add_edge(i, i, weight=1.)
            self.neg_word_graph.add_edge(i, i, weight=0.)  # 此处 weight 修改为 0

        # self.tfidf_graph = nx.Graph()  #

        if prefix == 'train':
            self.content = f"{processed_corpus_path}/train_only_remove_stopwords.txt"
        else:
            self.content = f"{processed_corpus_path}/allData_only_remove_stopwords.txt"  # test set 使用整个语料的图

        if dataset == '20ng':
            raw_corpus_path = f"data/{dataset}/20ng_all/{prefix}.jsonlist"  #
        else:
            raw_corpus_path = f"data/{dataset}/{prefix}.jsonlist"

        raw_corpus_dict = fh.read_jsonlist(raw_corpus_path)
        self.raw_corpus = []
        for item in raw_corpus_dict:
            self.raw_corpus.append(item['text'])

        self.get_tfidf_vec()
        self.build_word_graph(window_size, threshold_pos, threshold_neg)
        self.save(prefix)

    def build_word_graph(self, window_size=30, threshold_pos=0., threshold_neg=0.):
        """ threshold_pos and threshold_neg are both non-negative """
        pmi_edge_lst, self.pmi_time = get_pmi_edge(self.content, self.word2id, window_size=window_size)
        print("pmi time:", self.pmi_time)

        for word1, word2, weight in pmi_edge_lst:
            word_indx1 = self.word2id[word1]
            word_indx2 = self.word2id[word2]
            if word_indx1 == word_indx2:
                continue

            if weight > 0:
                if weight > threshold_pos:
                    self.pos_word_graph.add_edge(word_indx1, word_indx2, weight=weight)
            if weight < 0:
                weight = abs(weight)
                if weight > threshold_neg:
                    self.neg_word_graph.add_edge(word_indx1, word_indx2, weight=weight)

        print("details of positive word graph:")
        print_graph_detail(self.pos_word_graph)
        print("details of negative word graph:")
        print_graph_detail(self.neg_word_graph)

    def get_tfidf_vec(self):
        """
        学习获得tfidf矩阵，及其对应的单词序列
        """
        start = time()
        vectorizer = TfidfVectorizer(vocabulary=self.word2id, norm='l2', smooth_idf=False)
        tfidf = vectorizer.fit_transform(self.raw_corpus)

        self.tfidf_time = time() - start
        print("tfidf time:", self.tfidf_time)
        print("tfidf shape:", tfidf.shape)
        print("tfidf type:", type(tfidf))

        self.torch_sparse_tfidf = sparse_mx_to_torch_sparse_tensor(tfidf)

    def save(self, prefix):
        print("total time:", self.pmi_time + self.tfidf_time)
        nx.write_weighted_edgelist(self.pos_word_graph, f"{self.graph_path}/{prefix}.pos_word_graph.txt")
        nx.write_weighted_edgelist(self.neg_word_graph, f"{self.graph_path}/{prefix}.neg_word_graph.txt")
        # nx.write_weighted_edgelist(self.tfidf_graph, f"{self.graph_path}/{prefix}.tfidf_graph.txt")

        torch.save(self.torch_sparse_tfidf, f"{self.graph_path}/{prefix}.tfidf.pt")  # sparse_coo_tensor
        print("\n")


def main():
    dataset, window_size, threshold = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])

    BuildGraph(dataset, 'train', window_size=window_size, threshold_pos=threshold, threshold_neg=threshold)  # 可调整此处的窗口大小和阈值
    BuildGraph(dataset, 'test', window_size=window_size, threshold_pos=threshold, threshold_neg=threshold)

    train_pos_features, train_pos_adj = PrepareGraph(f"data/{dataset}/graph_{window_size}_{threshold}", 'train', positive=True)
    train_neg_features, train_neg_adj = PrepareGraph(f"data/{dataset}/graph_{window_size}_{threshold}", 'train', positive=False)
    train_graphs_20ng = (train_pos_features, train_pos_adj, train_neg_features, train_neg_adj)
    torch.save(train_graphs_20ng, f"data/{dataset}/graph_{window_size}_{threshold}/train_graphs.pt")
    test_pos_features, test_pos_adj = PrepareGraph(f"data/{dataset}/graph_{window_size}_{threshold}", 'test', positive=True)
    test_neg_features, test_neg_adj = PrepareGraph(f"data/{dataset}/graph_{window_size}_{threshold}", 'test', positive=False)
    test_graphs_20ng = (test_pos_features, test_pos_adj, test_neg_features, test_neg_adj)
    torch.save(test_graphs_20ng, f"data/{dataset}/graph_{window_size}_{threshold}/test_graphs.pt")

    # BuildGraph("imdb", 'train')
    # BuildGraph("imdb", 'test')

    # BuildGraph("wiki", 'train')
    # BuildGraph("wiki", 'test')

    # BuildGraph("nips", 'train')
    # BuildGraph("nips", 'test')


if __name__ == '__main__':
    setup_seed(42)
    main()
