from data import TextData
import json
from train import TextTrain
import csv
from utils import *
import torch
from sklearn.metrics import classification_report
import os
import time
import numpy as np


train_path = './data/train_data.jsonl'
test_path = './data/test_data.jsonl'
transfer_path = './data/transfer2.csv'
train_url_path = './data/train_url.csv'
test_url_path = './data/test_url.csv'


with open(train_path, 'r') as f:
    train_data = [json.loads(line) for line in f]
    train_content_list = [tmp['text'] for tmp in train_data]
    train_label_list = [tmp['label'] for tmp in train_data]

with open(test_path, 'r') as f:
    test_data = [json.loads(line) for line in f]
    test_content_list = [tmp['text'] for tmp in test_data]
    test_label_list = [tmp['label'] for tmp in test_data]

with open(transfer_path, 'r') as f:
    reader = csv.reader(f)
    rows = list(reader)
    transfer_content_list = []
    transfer_label_list = []
    for row in rows[1:]:
        transfer_content_list.append(row[1])
        if row[0] == 'ham' or row[0] == '0':
            transfer_label_list.append('0')
        else:
            transfer_label_list.append('1')

with open(train_url_path, 'r') as f:
    reader = csv.reader(f)
    rows = list(reader)
    train_urls = [row[0] for row in rows[1:]]
    train_labels = [row[1] for row in rows[1:]]

with open(test_url_path, 'r') as f:
    reader = csv.reader(f)
    rows = list(reader)
    test_urls = [row[0] for row in rows[1:]]
    test_labels = [row[1] for row in rows[1:]]


def get_config(model_type, td):
    if model_type == 'cnn':
        config = {
        'model_type': 'cnn',
        'num_iterations': 10000,
        'batch_size': 128,
        'learning_rate': 0.001,
        'print_per_batch': 100,

        'embedding_dim': 128,
        'num_filters': 512,
        'kernel_size': 5,
        'hidden_dim': 256,
        'dropout_keep_prob': 0.5,
        }
        config['vocab_size'] = td.vocab_size
        config['num_classes'] = td.num_classes
        config['labels'] = td.labels
        return config
    elif model_type == 'lstm':
        config = {
            'model_type': 'lstm',
            'num_iterations': 5000,  # 训练迭代次数
            'batch_size': 128,       # 批量大小
            'learning_rate': 0.001,  # 学习率
            'print_per_batch': 100,  # 每多少步打印一次
            'embedding_dim': 128,    # 词嵌入维度
            'hidden_dim': 256,       # LSTM 隐藏层维度
            'dropout_keep_prob': 0.5,# Dropout 保留概率
        }
        config['vocab_size'] = td.vocab_size
        config['num_classes'] = td.num_classes
        config['labels'] = td.labels
        return config
    else:
        raise ValueError('model_type must be cnn or lstm')


def train_body(model_type, train_content_list, train_label_list, test_content_list, test_label_list):
    td = TextData(train_content_list, train_label_list, test_content_list, test_label_list)
    train_X, train_Y, test_X, test_Y = td.get_data(type=model_type+'body', is_training=True)
    config = get_config(model_type, td)
    train = TextTrain(config, train_X, train_Y, test_X, test_Y)
    train.train()

def test_body(model_type, train_content_list, train_label_list, test_content_list, test_label_list):
    td = TextData(train_content_list, train_label_list, test_content_list, test_label_list)
    train_X, train_Y, test_X, test_Y = td.get_data(type=model_type+'body', is_training=False)
    config = get_config(model_type, td)
    train = TextTrain(config, train_X, train_Y, test_X, test_Y)
    train.test()

def transfer_body(model_type, train_content_list, train_label_list, transfer_content_list, transfer_label_list):
    td = TextData(train_content_list, train_label_list, transfer_content_list, transfer_label_list)
    train_X, train_Y, transfer_X, transfer_Y = td.get_data(type=model_type+'body', is_training=False)
    config = get_config(model_type, td)
    train = TextTrain(config, train_X, train_Y, transfer_X, transfer_Y)
    train.test()

def train_url(model_type, train_urls, train_labels, test_urls, test_labels):
    td = TextData(train_urls, train_labels, test_urls, test_labels, seq_length=200, is_url=True)
    train_X, train_Y, test_X, test_Y = td.get_data(type=model_type+'url', is_training=True)
    config = get_config(model_type, td)
    train = TextTrain(config, train_X, train_Y, test_X, test_Y, is_url=True)
    train.train()

def test_url(model_type, train_urls, train_labels, test_urls, test_labels):
    td = TextData(train_urls, train_labels, test_urls, test_labels, seq_length=200, is_url=True)
    train_X, train_Y, test_X, test_Y = td.get_data(type=model_type+'url', is_training=False)
    config = get_config(model_type, td)
    train = TextTrain(config, train_X, train_Y, test_X, test_Y, is_url=True)
    train.test()






# train_body('lstm', train_content_list, train_label_list, test_content_list, test_label_list) # 训练body
test_body('lstm', train_content_list, train_label_list, test_content_list, test_label_list) # 测试body
# transfer_body('lstm', train_content_list, train_label_list, transfer_content_list, transfer_label_list) # 测试body

# train_url('lstm', train_urls, train_labels, test_urls, test_labels) # 训练url
# test_url('lstm', train_urls, train_labels, test_urls, test_labels) # 测试url

# train_body('lstm', train_content_list, train_label_list, test_content_list, test_label_list) # 训练body
# test_body('lstm', train_content_list, train_label_list, test_content_list, test_label_list) # 测试body
# transfer_body('lstm', train_content_list, train_label_list, transfer_content_list, transfer_label_list)

# train_url('cnn', train_urls, train_labels, test_urls, test_labels) # 训练body
# test_url('cnn', train_urls, train_labels, test_urls, test_labels) # 测试body
