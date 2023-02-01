import os
import sys
from optparse import OptionParser

import gensim
import numpy as np
from numpy.linalg import norm
import pandas as pd

import file_handling as fh
from scholar import Scholar
import torch
import pickle
from graph_utils import load_graphs


def main(args):
    usage = "%prog input_dir"
    parser = OptionParser(usage=usage)
    parser.add_option('-k', dest='n_topics', type=int, default=20,
                      help='Size of latent representation (~num topics): default=%default')
    parser.add_option('-l', dest='learning_rate', type=float, default=0.002,
                      help='Initial learning rate: default=%default')
    parser.add_option('-m', dest='momentum', type=float, default=0.99,
                      help='beta1 for Adam: default=%default')
    parser.add_option('--batch-size', dest='batch_size', type=int, default=200,
                      help='Size of minibatches: default=%default')
    parser.add_option('--epochs', type=int, default=200,
                      help='Number of epochs: default=%default')
    parser.add_option('--train-prefix', type=str, default='train',
                      help='Prefix of train set: default=%default')
    parser.add_option('--test-prefix', type=str, default=None,
                      help='Prefix of test set: default=%default')
    parser.add_option('--labels', type=str, default=None,
                      help='Read labels from input_dir/[train|test].labels.csv: default=%default')
    parser.add_option('--prior-covars', type=str, default=None,
                      help='Read prior covariates from files with these names (comma-separated): default=%default')
    parser.add_option('--topic-covars', type=str, default=None,
                      help='Read topic covariates from files with these names (comma-separated): default=%default')
    parser.add_option('--interactions', action="store_true", default=False,
                      help='Use interactions between topics and topic covariates: default=%default')
    parser.add_option('--covars-predict', action="store_true", default=False,
                      help='Use covariates as input to classifier: default=%default')
    parser.add_option('--min-prior-covar-count', type=int, default=None,
                      help='Drop prior covariates with less than this many non-zero values in the training dataa: default=%default')
    parser.add_option('--min-topic-covar-count', type=int, default=None,
                      help='Drop topic covariates with less than this many non-zero values in the training dataa: default=%default')
    parser.add_option('-r', action="store_true", default=False,
                      help='Use default regularization: default=%default')
    parser.add_option('--l1-topics', type=float, default=0.0,
                      help='Regularization strength on topic weights: default=%default')
    parser.add_option('--l1-topic-covars', type=float, default=0.0,
                      help='Regularization strength on topic covariate weights: default=%default')
    parser.add_option('--l1-interactions', type=float, default=0.0,
                      help='Regularization strength on topic covariate interaction weights: default=%default')
    parser.add_option('--l2-prior-covars', type=float, default=0.0,
                      help='Regularization strength on prior covariate weights: default=%default')
    parser.add_option('--o', dest='output_dir', type=str, default='output',
                      help='Output directory: default=%default')
    parser.add_option('--emb-dim', type=int, default=300,
                      help='Dimension of input embeddings: default=%default')
    parser.add_option('--w2v', dest='word2vec_file', type=str, default=None,
                      help='Use this word2vec .bin file to initialize and fix embeddings: default=%default')
    parser.add_option('--alpha', type=float, default=1.0,
                      help='Hyperparameter for logistic normal prior: default=%default')
    # parser.add_option('--no-bg', action="store_true", default=False,
    #                   help='Do not use background freq: default=%default')
    parser.add_option('--dev-folds', type=int, default=0,
                      help='Number of dev folds: default=%default')
    parser.add_option('--dev-fold', type=int, default=0,               # 只取一折，取得是第几折
                      help='Fold to use as dev (if dev_folds > 0): default=%default')
    parser.add_option('--device', type=int, default=None,
                      help='GPU to use: default=%default')
    parser.add_option('--seed', type=int, default=None, help='Random seed: default=%default')
    parser.add_option('--dist', type=int, default=None, help='distance')

    parser.add_option('--model', type=str, default='scholar')
    parser.add_option('--topk', type=int, default=1)
    parser.add_option('--ablation', type=str, default=None)
    parser.add_option('--bg-freq', type=str, default='tfidf')

    options, args = parser.parse_args(args)
    input_dir = args[0]
    graph_path = args[1]  #
    options.input_dir = input_dir

    if '20ng' in input_dir:
        if 'contrastive' in options.model: torch.save(options, './weights/options/contrastive_20ng_50_options.pt')
        else: torch.save(options, './weights/options/20ng_50_options.pt')
        embeddings_dir = 'data/20ng/embeddings/'
        options.labels = 'group'  # 先加载 label，保存之后再令其为 None
    elif 'imdb' in input_dir:
        if 'contrastive' in options.model: torch.save(options, './weights/options/contrastive_imdb_50_options.pt')
        else: torch.save(options, './weights/options/imdb_50_options.pt')
        embeddings_dir = 'data/imdb/embeddings/'
        options.labels = 'sentiment'
    elif 'wiki' in input_dir:
        if 'contrastive' in options.model: torch.save(options, './weights/options/contrastive_wiki_50_options.pt')
        else: torch.save(options, './weights/options/wiki_50_options.pt')
        embeddings_dir = 'data/wiki/embeddings/'
    elif 'nips' in input_dir:
        if 'contrastive' in options.model: torch.save(options, './weights/options/contrastive_nips_50_options.pt')
        else: torch.save(options, './weights/options/nips_50_options.pt')
        embeddings_dir = 'data/nips/embeddings/'
    elif 'ag_news' in input_dir:
        if 'contrastive' in options.model: torch.save(options, './weights/options/contrastive_ag_news_50_options.pt')
        else: torch.save(options, './weights/options/ag_news_50_options.pt')
        embeddings_dir = 'data/ag_news/embeddings/'

    if options.r:
        options.l1_topics = 1.0
        options.l1_topic_covars = 1.0
        options.l1_interactions = 1.0

    if options.seed is not None:
        rng = np.random.RandomState(options.seed)
        seed = options.seed
    else:
        rng = np.random.RandomState(np.random.randint(0, 100000))
        seed = None

    # load the training data
    train_X_bow, vocab, row_selector, train_ids = load_word_counts(input_dir, options.train_prefix)
    train_X_tfidf = load_tfidf(graph_path, options.train_prefix, row_selector)  #
    train_bert_embeddings = load_bert_embeddings(embeddings_dir, options.train_prefix, row_selector)  #

    train_labels, label_type, label_names, n_labels = load_labels(input_dir, options.train_prefix, row_selector, options)
    train_prior_covars, prior_covar_selector, prior_covar_names, n_prior_covars = load_covariates(input_dir, options.train_prefix, row_selector, options.prior_covars, options.min_prior_covar_count)
    train_topic_covars, topic_covar_selector, topic_covar_names, n_topic_covars = load_covariates(input_dir, options.train_prefix, row_selector, options.topic_covars, options.min_topic_covar_count)
    options.n_train, vocab_size = train_X_bow.shape
    options.n_labels = n_labels

    if n_labels > 0:
        print("Train label proportions:", np.mean(train_labels, axis=0))

    # split into training and dev if desired
    train_indices, dev_indices = train_dev_split(options, rng)
    train_X_tfidf, dev_X_tfidf = split_matrix(train_X_tfidf, train_indices, dev_indices)  #
    train_X_bow, dev_X_bow = split_matrix(train_X_bow, train_indices, dev_indices)  #

    train_bert_embeddings, dev_bert_embeddings = split_matrix(train_bert_embeddings, train_indices, dev_indices)  #

    train_labels, dev_labels = split_matrix(train_labels, train_indices, dev_indices)   # train_labels 是 ndarray
    train_prior_covars, dev_prior_covars = split_matrix(train_prior_covars, train_indices, dev_indices)
    train_topic_covars, dev_topic_covars = split_matrix(train_topic_covars, train_indices, dev_indices)
    if dev_indices is not None:
        dev_ids = [train_ids[i] for i in dev_indices]
        train_ids = [train_ids[i] for i in train_indices]
    else:
        dev_ids = None

    n_train, _ = train_X_bow.shape

    # load the test data
    if options.test_prefix is not None:
        test_X_bow, _, row_selector, test_ids = load_word_counts(input_dir, options.test_prefix, vocab=vocab)
        test_X_tfidf = load_tfidf(graph_path, options.test_prefix, row_selector)  #
        test_bert_embeddings = load_bert_embeddings(embeddings_dir, options.test_prefix, row_selector)  #

        test_labels, _, _, _ = load_labels(input_dir, options.test_prefix, row_selector, options)
        test_prior_covars, _, _, _ = load_covariates(input_dir, options.test_prefix, row_selector, options.prior_covars, covariate_selector=prior_covar_selector)
        test_topic_covars, _, _, _ = load_covariates(input_dir, options.test_prefix, row_selector, options.topic_covars, covariate_selector=topic_covar_selector)
        n_test, _ = test_X_bow.shape

    else:
        test_X_tfidf = None
        test_X_bow = None
        test_bert_embeddings = None
        n_test = 0
        test_labels = None
        test_prior_covars = None
        test_topic_covars = None
        test_ids = None

    # make output directory
    fh.makedirs(options.output_dir)

    # 保存 labels, 然后令其为 None, 以免影响模型的训练
    np.save(os.path.join(options.output_dir, 'train_labels'), train_labels)
    np.save(os.path.join(options.output_dir, 'dev_labels'), dev_labels)
    np.save(os.path.join(options.output_dir, 'test_labels'), test_labels)
    train_labels = dev_labels = test_labels = None
    n_labels = 0

    # initialize the background using overall word frequencies
    if options.bg_freq == 'tfidf':
        init_bg = get_init_bg(train_X_tfidf, smooth=0.01)
    elif options.bg_freq == 'bow':
        init_bg = get_init_bg(train_X_bow)
    else:
        init_bg = np.zeros(len(vocab))

    # combine the network configuration parameters into a dictionary
    network_architecture = make_network(options, vocab_size, label_type, n_labels, n_prior_covars, n_topic_covars)

    print("Network architecture:")
    for key, val in network_architecture.items():
        print(key + ':', val)

    # load word vectors
    w2v_embeddings, update_w2v_embeddings = load_word_vectors(options, rng, vocab)

    # 加载 词共现图
    # train_graphs = load_graphs(graph_path, 'train')
    train_graphs = torch.load(f'{graph_path}/train_graphs.pt')

    if options.test_prefix is not None:
        # test_graphs = load_graphs(graph_path, 'test')
        test_graphs = torch.load(f'{graph_path}/test_graphs.pt')
    else:
        test_graphs = ()

    # create the model
    model = Scholar(network_architecture, alpha=options.alpha, learning_rate=options.learning_rate, init_embeddings=w2v_embeddings, update_embeddings=update_w2v_embeddings, init_bg=init_bg,
                    adam_beta1=options.momentum, device=options.device, seed=seed, classify_from_covars=options.covars_predict, model=options.model, topk=options.topk, ablation=options.ablation)

    vocabs = fh.read_json(os.path.join(options.input_dir, 'train.vocab.json'))
    token2id = {v: i for i, v in enumerate(vocabs)}
    with torch.no_grad():
        pos_cos, neg_cos = check_contras_sample(model, train_X_tfidf, train_X_bow, train_bert_embeddings,
                                                train_labels, train_prior_covars, train_topic_covars,
                                                options.batch_size, 1.0, train_graphs)
        k = 4
        train_X_tfidf = train_X_tfidf.numpy()
        print(train_X_tfidf[k, :])
        print(pos_cos[k, :])
        print('train: ', [vocabs[_] for _ in np.nonzero(train_X_tfidf[k, :])[0]])

        difference = neg_cos[k, :] - train_X_tfidf[k, :]
        topk_diff = np.argsort(difference)[-15:]
        print('difference: ', [vocabs[_] for _ in topk_diff])

        # clntm_negtive_sample = ['article', 'get', 'host', 'just', 'know', 'like', 'lines', 'nntp', 'one',
        #                        'organization', 'people', 'posting', 'subject', 'university', 'writes']
        gctm_neg_sample = ['hot', 'virginia', 'guns', 'absolutely', 'max', 'protect', 'paul', 'realize', 'opinions',
                           'nntp', 'posting', 'upon', 'hardly', 'attacks', 'university']
        # clntm_negtive_sample_ids = [token2id[_] for _ in clntm_negtive_sample]
        gctm_neg_sample_ids = [token2id[_] for _ in gctm_neg_sample]

        # print('clntm neg samples:', train_X_tfidf[k, clntm_negtive_sample_ids])
        print('gctm neg samples:', train_X_tfidf[k, gctm_neg_sample_ids])
    exit()
    # train the model
    print("Optimizing full model")
    model = train(model, network_architecture, train_X_tfidf, train_X_bow, train_bert_embeddings, train_labels, train_prior_covars, train_topic_covars, training_epochs=options.epochs, batch_size=options.batch_size,
                  rng=rng, dev_X_tfidf=dev_X_tfidf, dev_X_bow=dev_X_bow, bert_dev=dev_bert_embeddings, Y_dev=dev_labels, PC_dev=dev_prior_covars, TC_dev=dev_topic_covars, graphs=train_graphs)

    # display and save weights
    print_and_save_weights(options, model, vocab, prior_covar_names, topic_covar_names)

    # Evaluate perplexity on dev and test data
    if dev_X_bow is not None:
        perplexity = evaluate_perplexity(model, dev_X_tfidf, dev_X_bow, dev_bert_embeddings, dev_labels, dev_prior_covars, dev_topic_covars, options.batch_size, eta_bn_prop=0.0, graphs=train_graphs)
        print("Dev perplexity = %0.4f" % perplexity)
        fh.write_list_to_text([str(perplexity)], os.path.join(options.output_dir, 'perplexity.dev.txt'))

    if test_X_bow is not None:
        perplexity = evaluate_perplexity(model, test_X_tfidf, test_X_bow, test_bert_embeddings, test_labels, test_prior_covars, test_topic_covars, options.batch_size, eta_bn_prop=0.0, graphs=test_graphs)
        print("Test perplexity = %0.4f" % perplexity)
        fh.write_list_to_text([str(perplexity)], os.path.join(options.output_dir, 'perplexity.test.txt'))

    # evaluate accuracy on predicting labels
    if n_labels > 0:
        print("Predicting labels")
        predict_labels_and_evaluate(model, train_X_tfidf, train_X_bow, train_bert_embeddings, train_labels, train_prior_covars, train_topic_covars, options.output_dir, subset='train', graphs=train_graphs)

        if dev_X_bow is not None:
            predict_labels_and_evaluate(model, dev_X_tfidf, dev_X_bow, dev_bert_embeddings, dev_labels, dev_prior_covars, dev_topic_covars, options.output_dir, subset='dev')

        if test_X_bow is not None:
            predict_labels_and_evaluate(model, test_X_tfidf, test_X_bow, test_bert_embeddings, test_labels, test_prior_covars, test_topic_covars, options.output_dir, subset='test', graphs=test_graphs)

    # print label probabilities for each topic
    if n_labels > 0:
        print_topic_label_associations(options, label_names, model, n_prior_covars, n_topic_covars)

    # save document representations
    print("Saving document representations")
    save_document_representations(model, train_X_tfidf, train_X_bow, train_bert_embeddings, train_labels, train_prior_covars, train_topic_covars, train_ids, options.output_dir, 'train', batch_size=options.batch_size, graphs=train_graphs)

    if dev_X_bow is not None:
        save_document_representations(model, dev_X_tfidf, dev_X_bow, dev_bert_embeddings, dev_labels, dev_prior_covars, dev_topic_covars, dev_ids, options.output_dir, 'dev', batch_size=options.batch_size, graphs=train_graphs)

    if n_test > 0:
        save_document_representations(model, test_X_tfidf, test_X_bow, test_bert_embeddings, test_labels, test_prior_covars, test_topic_covars, test_ids, options.output_dir, 'test', batch_size=options.batch_size, graphs=test_graphs)


def load_tfidf(graph_path, input_prefix, row_selector):
    print("Loading data")
    # load the tfidf vector and convert to a dense tensor
    X_tfidf = torch.load(f"{graph_path}/{input_prefix}.tfidf.pt").to_dense()  # torch.tensor
    X_tfidf = X_tfidf[row_selector, :]

    return X_tfidf


def load_bert_embeddings(embeddings_dir, input_prefix, row_selector):
    # 加载 bert embeddings
    with open(embeddings_dir + input_prefix + '_embeddings.pkl', "rb") as fIn:
        stored_data = pickle.load(fIn)
        # train_sentences = stored_data['sentences']
        bert_embeddings = stored_data['embeddings']  # numpy array

    bert_embeddings = bert_embeddings[row_selector, :]

    return bert_embeddings


def load_word_counts(input_dir, input_prefix, vocab=None):
    print("Loading data")
    # laod the word counts and convert to a dense matrix
    # temp = fh.load_sparse(os.path.join(input_dir, input_prefix + '.npz')).todense()
    # X = np.array(temp, dtype='float32')
    X_bow = fh.load_sparse(os.path.join(input_dir, input_prefix + '.npz')).toarray()
    # load the vocabulary
    if vocab is None:
        vocab = fh.read_json(os.path.join(input_dir, input_prefix + '.vocab.json'))
    n_items, vocab_size = X_bow.shape
    assert vocab_size == len(vocab)
    print("Loaded %d documents with %d features" % (n_items, vocab_size))

    ids = fh.read_json(os.path.join(input_dir, input_prefix + '.ids.json'))

    # filter out empty documents and return a boolean selector for filtering labels and covariates
    row_sums = np.array(X_bow.sum(axis=1)).reshape((n_items,))
    row_selector = np.array(row_sums > 0, dtype=bool)

    print("Found %d non-empty documents" % np.sum(row_selector))
    X_bow = X_bow[row_selector, :]
    ids = [doc_id for i, doc_id in enumerate(ids) if row_selector[i]]

    return X_bow, vocab, row_selector, ids


def load_labels(input_dir, input_prefix, row_selector, options):
    labels = None
    label_type = None
    label_names = None
    n_labels = 0
    # load the label file if given
    if options.labels is not None:
        label_file = os.path.join(input_dir, input_prefix + '.' + options.labels + '.csv')
        if os.path.exists(label_file):
            print("Loading labels from", label_file)
            temp = pd.read_csv(label_file, header=0, index_col=0)
            label_names = temp.columns
            labels = np.array(temp.values)
            # select the rows that match the non-empty documents (from load_word_counts)
            labels = labels[row_selector, :]
            n, n_labels = labels.shape
            print("Found %d labels" % n_labels)
        else:
            raise(FileNotFoundError("Label file {:s} not found".format(label_file)))

    return labels, label_type, label_names, n_labels


def load_covariates(input_dir, input_prefix, row_selector, covars_to_load, min_count=None, covariate_selector=None):

    covariates = None
    covariate_names = None
    n_covariates = 0
    if covars_to_load is not None:
        covariate_list = []
        covariate_names_list = []
        covar_file_names = covars_to_load.split(',')
        # split the given covariate names by commas, and load each one
        for covar_file_name in covar_file_names:
            covariates_file = os.path.join(input_dir, input_prefix + '.' + covar_file_name + '.csv')
            if os.path.exists(covariates_file):
                print("Loading covariates from", covariates_file)
                temp = pd.read_csv(covariates_file, header=0, index_col=0)
                covariate_names = temp.columns
                covariates = np.array(temp.values, dtype=np.float32)
                # select the rows that match the non-empty documents (from load_word_counts)
                covariates = covariates[row_selector, :]
                covariate_list.append(covariates)
                covariate_names_list.extend(covariate_names)
            else:
                raise(FileNotFoundError("Covariates file {:s} not found".format(covariates_file)))

        # combine the separate covariates into a single matrix
        covariates = np.hstack(covariate_list)
        covariate_names = covariate_names_list

        _, n_covariates = covariates.shape

        # if a covariate_selector has been given (from a previous call of load_covariates), drop columns
        if covariate_selector is not None:
            covariates = covariates[:, covariate_selector]
            covariate_names = [name for i, name in enumerate(covariate_names) if covariate_selector[i]]
            n_covariates = len(covariate_names)
        # otherwise, choose which columns to drop based on how common they are (for binary covariates)
        elif min_count is not None and int(min_count) > 0:
            print("Removing rare covariates")
            covar_sums = covariates.sum(axis=0).reshape((n_covariates, ))
            covariate_selector = covar_sums > int(min_count)
            covariates = covariates[:, covariate_selector]
            covariate_names = [name for i, name in enumerate(covariate_names) if covariate_selector[i]]
            n_covariates = len(covariate_names)

    return covariates, covariate_selector, covariate_names, n_covariates


def train_dev_split(options, rng):
    # randomly split into train and dev
    if options.dev_folds > 0:
        n_dev = int(options.n_train / options.dev_folds)
        indices = np.array(range(options.n_train), dtype=int)
        rng.shuffle(indices)
        if options.dev_fold < options.dev_folds - 1:
            dev_indices = indices[n_dev * options.dev_fold: n_dev * (options.dev_fold +1)]
        else:
            dev_indices = indices[n_dev * options.dev_fold:]
        train_indices = list(set(indices) - set(dev_indices))
        print('number of train docs:', len(train_indices), 'number of valid docs:', len(dev_indices))
        return train_indices, dev_indices

    else:
        return None, None


def split_matrix(train_X, train_indices, dev_indices):
    # split a matrix (word counts, labels, or covariates), into train and dev
    if train_X is not None and dev_indices is not None:
        dev_X = train_X[dev_indices, :]
        train_X = train_X[train_indices, :]
        return train_X, dev_X
    else:
        return train_X, None


def get_init_bg(data, smooth=1.0):
    # Compute the log background frequency of all words
    # sums = np.sum(data, axis=0)+1
    n_items, vocab_size = data.shape
    sums = np.array(data.sum(axis=0)).reshape((vocab_size,)) + smooth
    print("Computing background frequencies")
    print("Min/max word counts in training data: %d %d" % (int(np.min(sums)), int(np.max(sums))))
    bg = np.array(np.log(sums) - np.log(float(np.sum(sums))), dtype=np.float32)  # 词表中每个词的对数词频
    return bg


def load_word_vectors(options, rng, vocab):
    # load word2vec vectors if given
    if options.word2vec_file is not None:
        vocab_size = len(vocab)
        vocab_dict = dict(zip(vocab, range(vocab_size)))
        # randomly initialize word vectors for each term in the vocabualry
        embeddings = np.array(rng.rand(options.emb_dim, vocab_size) * 0.25 - 0.5, dtype=np.float32)
        count = 0
        print("Loading word vectors")
        # load the word2vec vectors
        pretrained = gensim.models.KeyedVectors.load_word2vec_format(options.word2vec_file, binary=True)

        # replace the randomly initialized vectors with the word2vec ones for any that are available
        for word, index in vocab_dict.items():
            if word in pretrained:
                count += 1
                embeddings[:, index] = pretrained[word]

        print("Found embeddings for %d words" % count)
        update_embeddings = False
    else:
        embeddings = None
        update_embeddings = True

    return embeddings, update_embeddings


def make_network(options, vocab_size, label_type=None, n_labels=0, n_prior_covars=0, n_topic_covars=0):
    # Assemble the network configuration parameters into a dictionary
    network_architecture = \
        dict(embedding_dim=options.emb_dim,
             n_topics=options.n_topics,
             vocab_size=vocab_size,
             label_type=label_type,
             n_labels=n_labels,
             n_prior_covars=n_prior_covars,
             n_topic_covars=n_topic_covars,
             l1_beta_reg=options.l1_topics,
             l1_beta_c_reg=options.l1_topic_covars,
             l1_beta_ci_reg=options.l1_interactions,
             l2_prior_reg=options.l2_prior_covars,
             classifier_layers=1,
             use_interactions=options.interactions,
             dist=options.dist,
             model=options.model
             )
    return network_architecture


def train(model, network_architecture, train_X_tfidf, train_X_bow, train_bert_embeddings, Y, PC, TC, batch_size=200, training_epochs=100, display_step=5, dev_X_tfidf=None,
          dev_X_bow=None, bert_dev=None, Y_dev=None, PC_dev=None, TC_dev=None, bn_anneal=True, init_eta_bn_prop=1.0, rng=None, min_weights_sq=1e-7, graphs=None):
    # Train the model
    n_train, vocab_size = train_X_bow.shape
    mb_gen = create_minibatch(train_X_tfidf, train_X_bow, train_bert_embeddings, Y, PC, TC, batch_size=batch_size, rng=rng)
    total_batch = int(n_train / batch_size)
    batches = 0
    eta_bn_prop = init_eta_bn_prop  # interpolation between batch norm and no batch norm in final layer of recon

    model.train()

    n_topics = network_architecture['n_topics']
    n_topic_covars = network_architecture['n_topic_covars']
    vocab_size = network_architecture['vocab_size']

    # create matrices to track the current estimates of the priors on the individual weights
    if network_architecture['l1_beta_reg'] > 0:
        l1_beta = 0.5 * np.ones([vocab_size, n_topics], dtype=np.float32) / float(n_train)
    else:
        l1_beta = None

    if network_architecture['l1_beta_c_reg'] > 0 and network_architecture['n_topic_covars'] > 0:
        l1_beta_c = 0.5 * np.ones([vocab_size, n_topic_covars], dtype=np.float32) / float(n_train)
    else:
        l1_beta_c = None

    if network_architecture['l1_beta_ci_reg'] > 0 and network_architecture['n_topic_covars'] > 0 and network_architecture['use_interactions']:
        l1_beta_ci = 0.5 * np.ones([vocab_size, n_topics * n_topic_covars], dtype=np.float32) / float(n_train)
    else:
        l1_beta_ci = None

    # Training cycle
    for epoch in range(1, training_epochs + 1):
        avg_cost = 0.
        accuracy = 0.
        avg_nl = 0.
        avg_kld = 0.
        avg_contrast_loss = 0.
        # Loop over all batches
        for i in range(total_batch):
            # get a minibatch
            batch_x_tfidf, batch_x_bow, batch_bert_embeddings, batch_ys, batch_pcs, batch_tcs = next(mb_gen)
            # do one minibatch update
            cost, recon_y, thetas, nl, kld, contrast_loss = model.fit(batch_x_tfidf, batch_x_bow, batch_bert_embeddings, batch_ys, batch_pcs, batch_tcs,
                                                                      eta_bn_prop=eta_bn_prop, l1_beta=l1_beta, l1_beta_c=l1_beta_c, l1_beta_ci=l1_beta_ci, graphs=graphs)

            # compute accuracy on minibatch
            if network_architecture['n_labels'] > 0:
                accuracy += np.sum(np.argmax(recon_y, axis=1) == np.argmax(batch_ys, axis=1)) / float(n_train)

            # Compute average loss
            avg_cost += float(cost) / n_train * batch_size
            avg_nl += float(nl) / n_train * batch_size
            avg_kld += float(kld) / n_train * batch_size
            avg_contrast_loss += float(contrast_loss) / n_train * batch_size
            batches += 1
            if np.isnan(avg_cost):
                print(epoch, i, np.sum(batch_x_bow, 1).astype(np.int), batch_x_bow.shape)
                print('Encountered NaN, stopping training. Please check the learning_rate settings and the momentum.')
                sys.exit()

        # if we're using regularization, update the priors on the individual weights
        if network_architecture['l1_beta_reg'] > 0:
            weights = model.get_weights().T
            weights_sq = weights ** 2
            # avoid infinite regularization
            weights_sq[weights_sq < min_weights_sq] = min_weights_sq
            l1_beta = 0.5 / weights_sq / float(n_train)

        if network_architecture['l1_beta_c_reg'] > 0 and network_architecture['n_topic_covars'] > 0:
            weights = model.get_covar_weights().T
            weights_sq = weights ** 2
            weights_sq[weights_sq < min_weights_sq] = min_weights_sq
            l1_beta_c = 0.5 / weights_sq / float(n_train)

        if network_architecture['l1_beta_ci_reg'] > 0 and network_architecture['n_topic_covars'] > 0 and network_architecture['use_interactions']:
            weights = model.get_covar_interaction_weights().T
            weights_sq = weights ** 2
            weights_sq[weights_sq < min_weights_sq] = min_weights_sq
            l1_beta_ci = 0.5 / weights_sq / float(n_train)

        # Display logs per epoch step
        if epoch % display_step == 0:
            if network_architecture['n_labels'] > 0:
                print("Epoch:", '%3d' % epoch, "; cost =", "{:.9f}".format(avg_cost), "; training accuracy (noisy) =", "{:.9f}".format(accuracy))
            else:
                print("Epoch:", '%3d' % epoch, "cost=", "{:.9f}".format(avg_cost), "contrast loss=", "{:.9f}".format(avg_contrast_loss))

            if dev_X_bow is not None:
                # switch to eval mode for intermediate evaluation
                model.eval()
                dev_perplexity = evaluate_perplexity(model, dev_X_tfidf, dev_X_bow, bert_dev, Y_dev, PC_dev, TC_dev, batch_size, eta_bn_prop=eta_bn_prop, graphs=graphs)
                n_dev, _ = dev_X_bow.shape
                if network_architecture['n_labels'] > 0:
                    dev_pred_probs = predict_label_probs(model, dev_X_tfidf, dev_X_bow, bert_dev, PC_dev, TC_dev, eta_bn_prop=eta_bn_prop)
                    dev_predictions = np.argmax(dev_pred_probs, axis=1)
                    dev_accuracy = float(np.sum(dev_predictions == np.argmax(Y_dev, axis=1))) / float(n_dev)
                    print("Epoch: %3d; Dev perplexity = %0.4f; Dev accuracy = %0.4f" % (epoch, dev_perplexity, dev_accuracy))
                else:
                    print("Epoch: %3d; Dev perplexity = %0.4f" % (epoch, dev_perplexity))
                # switch back to training mode
                model.train()

        # anneal eta_bn_prop from 1.0 to 0.0 over training
        if bn_anneal:
            if eta_bn_prop > 0:
                eta_bn_prop -= 1.0 / float(0.75 * training_epochs)
                if eta_bn_prop < 0:
                    eta_bn_prop = 0.0

    # finish training
    model.eval()
    return model


def create_minibatch(X_tfidf, X_bow, bert_embeddings, Y, PC, TC, batch_size=200, rng=None):
    # Yield a random minibatch
    while True:
        # Return random data samples of a size 'minibatch_size' at each iteration
        if rng is not None:
            ixs = rng.randint(X_bow.shape[0], size=batch_size)
        else:
            ixs = np.random.randint(X_bow.shape[0], size=batch_size)

        X_mb_tfidf = X_tfidf[ixs, :]  #
        X_mb_bow = X_bow[ixs, :].astype('float32')  #
        bert_embeddings_mb = bert_embeddings[ixs, :]  #

        if Y is not None:
            Y_mb = Y[ixs, :].astype('float32')
        else:
            Y_mb = None

        if PC is not None:
            PC_mb = PC[ixs, :].astype('float32')
        else:
            PC_mb = None

        if TC is not None:
            TC_mb = TC[ixs, :].astype('float32')
        else:
            TC_mb = None

        yield X_mb_tfidf, X_mb_bow, bert_embeddings_mb, Y_mb, PC_mb, TC_mb


def get_minibatch(X_tfidf, X_bow, bert_embeddings, Y, PC, TC, batch, batch_size=200):
    # Get a particular non-random segment of the data
    n_items, _ = X_bow.shape
    n_batches = int(np.ceil(n_items / float(batch_size)))
    if batch < n_batches - 1:
        ixs = np.arange(batch * batch_size, (batch + 1) * batch_size)
    else:
        ixs = np.arange(batch * batch_size, n_items)

    X_mb_tfidf = X_tfidf[ixs, :]  # 输入是 tfidf，使用这个
    X_mb_bow = X_bow[ixs, :].astype('float32')  # 如果输入是 BoW，则使用这个
    bert_embeddings_mb = bert_embeddings[ixs, :].astype('float32')   #

    if Y is not None:
        Y_mb = Y[ixs, :].astype('float32')
    else:
        Y_mb = None

    if PC is not None:
        PC_mb = PC[ixs, :].astype('float32')
    else:
        PC_mb = None

    if TC is not None:
        TC_mb = TC[ixs, :].astype('float32')
    else:
        TC_mb = None

    return X_mb_tfidf, X_mb_bow, bert_embeddings_mb, Y_mb, PC_mb, TC_mb


def predict_label_probs(model, X_tfidf, X_bow, bert_embeddings, PC, TC, batch_size=200, eta_bn_prop=0.0, graphs=None):
    # Predict a probability distribution over labels for each instance using the classifier part of the network

    n_items, _ = X_bow.shape
    n_batches = int(np.ceil(n_items / batch_size))
    pred_probs_all = []

    # make predictions on minibatches and then combine
    for i in range(n_batches):
        batch_x_tfidf, batch_x_bow, batch_bert_embeddings, batch_ys, batch_pcs, batch_tcs = get_minibatch(X_tfidf, X_bow, bert_embeddings, None, PC, TC, i, batch_size)
        Z, pred_probs = model.predict(batch_x_tfidf, batch_x_bow, batch_bert_embeddings, batch_pcs, batch_tcs, eta_bn_prop=eta_bn_prop, graphs=graphs)
        pred_probs_all.append(pred_probs)

    pred_probs = np.vstack(pred_probs_all)

    return pred_probs


def print_and_save_weights(options, model, vocab, prior_covar_names=None, topic_covar_names=None):

    # print background
    bg = model.get_bg()
    if options.bg_freq is not None:
        print_top_bg(bg, vocab)

    if '20ng' in options.input_dir:
        if 'contrastive' in options.model.lower():
            torch.save(model._model.state_dict(), './weights/contrastive_20ng_50.pt')
        else: torch.save(model._model.state_dict(), './weights/20ng_50.pt')
    elif 'imdb' in options.input_dir:
        if 'contrastive' in options.model.lower():
            torch.save(model._model.state_dict(), './weights/contrastive_imdb_50.pt')
        else: torch.save(model._model.state_dict(), './weights/imdb_50.pt')
    elif 'wiki' in options.input_dir:
        if 'contrastive' in options.model.lower():
            torch.save(model._model.state_dict(), './weights/contrastive_wiki_50.pt')
        else: torch.save(model._model.state_dict(), './weights/wiki_50.pt')
    elif 'nips' in options.input_dir:
        if 'contrastive' in options.model.lower():
            torch.save(model._model.state_dict(), './weights/contrastive_nips_50.pt')
        else: torch.save(model._model.state_dict(), './weights/nips_50.pt')
    elif 'ag_news' in options.input_dir:
        if 'contrastive' in options.model.lower():
            torch.save(model.state_dict(), './weights/contrastive_ag_news_50.pt')
        else: torch.save(model.state_dict(), './weights/ag_news_50.pt')

    # print topics
    emb = model.get_weights()
    # emb = model.get_beta().to('cpu').detach().numpy()
    print("Topics:")
    maw, sparsity = print_top_words(emb, vocab)
    print("sparsity in topics = %0.4f" % sparsity)
    save_weights(options.output_dir, emb, bg, vocab, sparsity_threshold=1e-5)

    fh.write_list_to_text(['{:.4f}'.format(maw)], os.path.join(options.output_dir, 'maw.txt'))
    fh.write_list_to_text(['{:.4f}'.format(sparsity)], os.path.join(options.output_dir, 'sparsity.txt'))

    if prior_covar_names is not None:
        prior_weights = model.get_prior_weights()
        print("Topic prior associations:")
        print("Covariates:", ' '.join(prior_covar_names))
        for k in range(options.n_topics):
            output = str(k) + ': '
            for c in range(len(prior_covar_names)):
                output += '%.4f ' % prior_weights[c, k]
            print(output)
        if options.output_dir is not None:
            np.savez(os.path.join(options.output_dir, 'prior_w.npz'), weights=prior_weights, names=prior_covar_names)

    if topic_covar_names is not None:
        beta_c = model.get_covar_weights()
        print("Covariate deviations:")
        maw, sparsity = print_top_words(beta_c, vocab, topic_covar_names)
        print("sparsity in covariates = %0.4f" % sparsity)
        if options.output_dir is not None:
            np.savez(os.path.join(options.output_dir, 'beta_c.npz'), beta=beta_c, names=topic_covar_names)

        if options.interactions:
            print("Covariate interactions")
            beta_ci = model.get_covar_interaction_weights()
            print(beta_ci.shape)
            if topic_covar_names is not None:
                names = [str(k) + ':' + c for k in range(options.n_topics) for c in topic_covar_names]
            else:
                names = None
            maw, sparsity = print_top_words(beta_ci, vocab, names)
            if options.output_dir is not None:
                np.savez(os.path.join(options.output_dir, 'beta_ci.npz'), beta=beta_ci, names=names)
            print("sparsity in covariate interactions = %0.4f" % sparsity)


def print_top_words(beta, feature_names, topic_names=None, n_pos=8, n_neg=8, sparsity_threshold=1e-5, values=False):
    """
    Display the highest and lowest weighted words in each topic, along with mean ave weight and sparisty
    """
    sparsity_vals = []
    maw_vals = []
    for i in range(len(beta)):
        # sort the beta weights
        order = list(np.argsort(beta[i]))
        order.reverse()
        output = ''
        # get the top words
        for j in range(n_pos):
            if np.abs(beta[i][order[j]]) > sparsity_threshold:
                output += feature_names[order[j]] + ' '
                if values:
                    output += '(' + str(beta[i][order[j]]) + ') '

        order.reverse()
        if n_neg > 0:
            output += ' / '
        # get the bottom words
        for j in range(n_neg):
            if np.abs(beta[i][order[j]]) > sparsity_threshold:
                output += feature_names[order[j]] + ' '
                if values:
                    output += '(' + str(beta[i][order[j]]) + ') '

        # compute sparsity
        sparsity = float(np.sum(np.abs(beta[i]) < sparsity_threshold) / float(len(beta[i])))
        maw = np.mean(np.abs(beta[i]))
        sparsity_vals.append(sparsity)
        maw_vals.append(maw)
        output += '; sparsity=%0.4f' % sparsity

        # print the topic summary
        if topic_names is not None:
            output = topic_names[i] + ': ' + output
        else:
            output = str(i) + ': ' + output
        print(output)

    # return mean average weight and sparsity
    return np.mean(maw_vals), np.mean(sparsity_vals)


def print_top_bg(bg, feature_names, n_top_words=10):
    # Print the most highly weighted words in the background log frequency
    print('Background frequencies of top words:')
    print(" ".join([feature_names[j]
                    for j in bg.argsort()[:-n_top_words - 1:-1]]))
    temp = bg.copy()
    temp.sort()
    print(np.exp(temp[:-n_top_words-1:-1]))


def check_contras_sample(model, X_tfidf, X_bow, bert_embeddings, Y, PC, TC, batch_size, eta_bn_prop, graphs=None):
    # Evaluate the approximate perplexity on a subset of the data (using words, labels, and covariates)
    model.eval()
    n_items, vocab_size = X_tfidf.shape
    doc_sums = np.array(X_tfidf.sum(axis=1), dtype=float).reshape((n_items,))
    if Y is not None:
        Y = Y.astype('float32')
    if PC is not None:
        PC = PC.astype('float32')
    if TC is not None:
        TC = TC.astype('float32')
    losses = []

    n_items, _ = X_tfidf.shape
    n_batches = int(np.ceil(n_items / batch_size))
    neg_cos = []
    pos_cos = []
    for i in range(n_batches):
        batch_x_tfidf, batch_x_bow, batch_bert_embeddings, batch_ys, batch_pcs, batch_tcs = get_minibatch(X_tfidf,
                                                                                                          X_bow,
                                                                                                          bert_embeddings,
                                                                                                          Y, PC, TC, i,
                                                                                                          batch_size)
        _, new_X, pos_X, neg_X = model.predict(batch_x_tfidf, batch_x_bow, batch_bert_embeddings, PC, TC, eta_bn_prop=eta_bn_prop, graphs=graphs)
        neg_cos.append(neg_X)
        pos_cos.append(pos_X)
    neg_cos = np.vstack(neg_cos)
    pos_cos = np.vstack(pos_cos)
    return pos_cos, neg_cos


def evaluate_perplexity(model, X_tfidf, X_bow, bert_embeddings, Y, PC, TC, batch_size, eta_bn_prop=0.0, graphs=None):
    # Evaluate the approximate perplexity on a subset of the data (using words, labels, and covariates)
    n_items, vocab_size = X_bow.shape
    doc_sums = np.array(X_bow.sum(axis=1), dtype=float).reshape((n_items,))

    if Y is not None:
        Y = Y.astype('float32')
    if PC is not None:
        PC = PC.astype('float32')
    if TC is not None:
        TC = TC.astype('float32')
    losses = []

    n_batches = int(np.ceil(n_items / batch_size))
    for i in range(n_batches):
        batch_x_tfidf, batch_x_bow, batch_bert_embeddings, batch_ys, batch_pcs, batch_tcs = get_minibatch(X_tfidf, X_bow, bert_embeddings, Y, PC, TC, i, batch_size)
        batch_losses = model.get_losses(batch_x_tfidf, batch_x_bow, batch_bert_embeddings, batch_ys, batch_pcs, batch_tcs, eta_bn_prop=eta_bn_prop, graphs=graphs)
        losses.append(batch_losses)
    losses = np.hstack(losses)
    perplexity = np.exp(np.mean(losses / doc_sums))
    return perplexity


def save_weights(output_dir, beta, bg, feature_names, sparsity_threshold=1e-5):
    # Save model weights to npz files (also the top words in each topic
    np.savez(os.path.join(output_dir, 'beta.npz'), beta=beta)
    if bg is not None:
        np.savez(os.path.join(output_dir, 'bg.npz'), bg=bg)
    fh.write_to_json(feature_names, os.path.join(output_dir, 'vocab.json'), sort_keys=False)

    topics_file = os.path.join(output_dir, 'topics.txt')
    lines = []
    for i in range(len(beta)):
        order = list(np.argsort(beta[i]))
        order.reverse()
        pos_words = [feature_names[j] for j in order[:100] if beta[i][j] > sparsity_threshold]
        output = ' '.join(pos_words)
        lines.append(output)

    fh.write_list_to_text(lines, topics_file)


def predict_labels_and_evaluate(model, X_tfidf, X_bow, bert_embeddings, Y, PC, TC, output_dir=None, subset='train', batch_size=200, graphs=None):
    # Predict labels for all instances using the classifier network and evaluate the accuracy
    pred_probs = predict_label_probs(model, X_tfidf, X_bow, bert_embeddings, PC, TC, batch_size, eta_bn_prop=0.0, graphs=graphs)
    np.savez(os.path.join(output_dir, 'pred_probs.' + subset + '.npz'), pred_probs=pred_probs)
    predictions = np.argmax(pred_probs, axis=1)
    accuracy = float(np.sum(predictions == np.argmax(Y, axis=1)) / float(len(Y)))
    print(subset, "accuracy on labels = %0.4f" % accuracy)
    if output_dir is not None:
        fh.write_list_to_text([str(accuracy)], os.path.join(output_dir, 'accuracy.' + subset + '.txt'))


def print_topic_label_associations(options, label_names, model, n_prior_covars, n_topic_covars):
    # Print associations between topics and labels
    if options.n_labels > 0 and options.n_labels < 7:
        print("Label probabilities based on topics")
        print("Labels:", ' '.join([name for name in label_names]))
    probs_list = []
    for k in range(options.n_topics):
        Z = np.zeros([1, options.n_topics]).astype('float32')
        Z[0, k] = 1.0
        Y = None
        if n_prior_covars > 0:
            PC = np.zeros([1, n_prior_covars]).astype('float32')
        else:
            PC = None
        if n_topic_covars > 0:
            TC = np.zeros([1, n_topic_covars]).astype('float32')
        else:
            TC = None

        probs = model.predict_from_topics(Z, PC, TC)
        probs_list.append(probs)
        if options.n_labels > 0 and options.n_labels < 7:
            output = str(k) + ': '
            for i in range(options.n_labels):
                output += '%.4f ' % probs[0, i]
            print(output)

    probs = np.vstack(probs_list)
    np.savez(os.path.join(options.output_dir, 'topics_to_labels.npz'), probs=probs, label=label_names)


def save_document_representations(model, X_tfidf, X_bow, bert_embeddings, Y, PC, TC, ids, output_dir, partition, batch_size=200, graphs=None):
    # compute the mean of the posterior of the latent representation for each documetn and save it
    if Y is not None:
        Y = np.zeros_like(Y)

    n_items, _ = X_bow.shape
    n_batches = int(np.ceil(n_items / batch_size))
    thetas = []

    for i in range(n_batches):
        batch_x_tfidf, batch_x_bow, batch_bert_embeddings, batch_ys, batch_pcs, batch_tcs = get_minibatch(X_tfidf, X_bow, bert_embeddings, Y, PC, TC, i, batch_size)
        thetas.append(model.compute_theta(batch_x_tfidf, batch_x_bow, batch_bert_embeddings, batch_ys, batch_pcs, batch_tcs, graphs=graphs))
    theta = np.vstack(thetas)

    np.savez(os.path.join(output_dir, 'theta.' + partition + '.npz'), theta=theta, ids=ids)


if __name__ == '__main__':
    main(sys.argv[1:])

