from collections import Counter
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re

# 确保在开始时下载 nltk 数据
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')


class TextData:
    # 将输入的文本和标签列表分为训练集和测试集
    # self, train_content_list, train_label_list, test_content_list, test_label_list, seq_length=5000
    # 初始化一些属性
    def __init__(
            self,
            train_content_list,
            train_label_list,
            test_content_list,
            test_label_list,
            seq_length=5000,
            is_url=False
    ):
        self.tokenizer = RegexpTokenizer(r'\w+')
        train_X, test_X, train_y, test_y = train_content_list, test_content_list, train_label_list, test_label_list
        self.train_content_list = train_X
        self.train_label_list = train_y
        self.test_content_list = test_X
        self.test_label_list = test_y
        self.content_list = self.train_content_list + self.test_content_list
        self.num_classes = len(np.unique(self.train_label_list + self.test_label_list))
        self.embedding_dim = 64
        self.seq_length = seq_length
        self.stop_words = set(stopwords.words('english'))
        self.is_training = True
        self.is_url = is_url

    
    def __process_url(self, url):
        url = url.lower()
        url = re.sub(r'https?://', '', url)
        url = re.sub(r'[/\-_=?&\.]', ' ', url)
        tokens = []
        for token in url.split():
            if len(token) > 20:
                tokens.extend(list(token))
            else:
                tokens.append(token)
        tokens = [
            token for token in tokens
            if token and not token.isnumeric()
        ]
        return tokens


    def __preprocess_text(self, text):
        if self.is_url:
            return self.__process_url(text)
        else:
            text = text.lower()
            tokens = self.tokenizer.tokenize(text)
            tokens = [
                word for word in tokens
                if word not in self.stop_words
                and not word.isnumeric()
                and len(word) > 1
            ]
            return tokens

    def __prepare_data(self, type):
        if self.is_training:
            all_tokens = []
            for content in self.content_list:
                tokens = self.__preprocess_text(content)
                all_tokens.extend(tokens)
            
            counter = Counter(all_tokens)
            vocabulary_list = ['PAD', 'UNK'] + [k for k, v in counter.most_common() if v > 1]
            self.vocab_size = len(vocabulary_list)
            self.word2id_dict = {word: idx for idx, word in enumerate(vocabulary_list)}
            torch.save(self.word2id_dict, f'./vocab/{type}_vocab.pkl')
        else:
            self.word2id_dict = torch.load(f'./vocab/{type}_vocab.pkl')
            self.vocab_size = len(self.word2id_dict)

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.train_label_list)
        self.labels = self.label_encoder.classes_

    def __content2X(self, content_list):
        # 使用优化后的预处理方法处理文本
        idlist_list = []
        for content in content_list:
            tokens = self.__preprocess_text(content)
            ids = [
                self.word2id_dict.get(token, self.word2id_dict['UNK'])
                for token in tokens
            ]
            idlist_list.append(ids)
        
        # 使用 torch 的 pad_sequence 进行填充
        padded_sequences = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(seq, dtype=torch.long) for seq in idlist_list],
            batch_first=True,
            padding_value=self.word2id_dict['PAD']
        )
        
        # 截断或填充到固定长度
        if padded_sequences.size(1) > self.seq_length:
            X = padded_sequences[:, :self.seq_length]
        else:
            padding = torch.full((padded_sequences.size(0), self.seq_length - padded_sequences.size(1)), self.word2id_dict['PAD'], dtype=torch.long)
            X = torch.cat([padding, padded_sequences], dim=1)
        return X.numpy()

    def __label2Y(self, label_list):
        return self.label_encoder.transform(label_list)

    def get_data(self, type, is_training=True):
        self.is_training = is_training
        self.__prepare_data(type)
        return (
            self.__content2X(self.train_content_list),
            self.__label2Y(self.train_label_list),
            self.__content2X(self.test_content_list),
            self.__label2Y(self.test_label_list)
        )
