import pickle
from sentence_transformers import SentenceTransformer
from file_handling import read_jsonlist
import os, sys


def get_bert_embeddings(dataset, device=0):
    """
    :param dataset: 20ng, imdb, nips
    :param device: int, GPU id
    """
    print(f'==> 目前处理的数据集是: {dataset} <==')
    if dataset == '20ng':
        dataset_dir = 'data/20ng/20ng_all'
        embeddings_dir = 'data/20ng/embeddings'
    else:
        dataset_dir = f'data/{dataset}'
        embeddings_dir = f'{dataset_dir}/embeddings'

    train_items = read_jsonlist(f"{dataset_dir}/train.jsonlist")
    test_items = read_jsonlist(f"{dataset_dir}/test.jsonlist")

    bert = SentenceTransformer("stsb-roberta-large", device=f'cuda:{device}')  # 此处修改设备

    bert.max_seq_length = 512

    train_texts = [train['text'] for train in train_items]
    test_texts = [test['text'] for test in test_items]

    print("encoding train set:")
    train_bert_embeddings = bert.encode(train_texts, show_progress_bar=True, batch_size=32)
    print("encoding test set:")
    test_bert_embeddings = bert.encode(test_texts, show_progress_bar=True, batch_size=32)

    if not os.path.exists(embeddings_dir):
        os.makedirs(embeddings_dir)

    # Store sentences & embeddings on disc
    save_embeddings(embeddings_dir, train_texts, train_bert_embeddings, 'train')
    save_embeddings(embeddings_dir, test_texts, test_bert_embeddings, 'test')


def save_embeddings(embeddings_dir, texts, embeddings, prefix):
    with open(f'{embeddings_dir}/{prefix}_embeddings.pkl', "wb") as fOut:
        pickle.dump({'sentences': texts, 'embeddings': embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'{prefix} embeddings size: ', embeddings.shape)


if __name__ == "__main__":
    dataset = sys.argv[1]
    get_bert_embeddings(dataset, device=3)
    # get_bert_embeddings("imdb")
    # get_bert_embeddings("wiki")
    # get_bert_embeddings("nips")
    # get_bert_embeddings("ag_news")
