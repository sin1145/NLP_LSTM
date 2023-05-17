import os
import jieba
import torch
from torch import nn
# 字典类，使词和id可相互查找
class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def __len__(self):
        return len(self.word2idx)

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx  
            self.idx2word[self.idx] = word  
            self.idx += 1

# 读取文件并预处理，利用Dictionary进行编码解码
class Corpus(object):
    def __init__(self):
        self.dictionary = Dictionary()  
        self.file_list = []

    def get_file(self, filepath):
        for root, path, fil in os.walk(filepath):
            for txt_file in fil:
                file_name = os.path.join(root,txt_file)
                self.file_list.append(file_name)  
        return self.file_list

    def get_data(self, batch_size):  
        ids = []
        for path in self.file_list:
            print(path)
            with open(path, 'r', encoding="ANSI") as f:
                for line in f.readlines():
                    line = line.replace(' ', '').replace('\u3000', '').replace('\t', '')
                    words = jieba.lcut(line) + ['<eos>']
                    for word in words: 
                        self.dictionary.add_word(word)
                        ids.extend([self.dictionary.word2idx[word]])
        ids = torch.LongTensor(ids)
        # batchsize矩阵重构
        num_batches = ids.size(0) // batch_size
        ids = ids[:num_batches * batch_size]
        ids = ids.view(batch_size, -1)
        return ids

class LSTMmodel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(LSTMmodel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)  
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)  
        self.linear = nn.Linear(hidden_size, vocab_size)  

    def forward(self, x, h):
        x = self.embed(x)
        out, (h, c) = self.lstm(x, h)
        out = out.reshape(out.size(0) * out.size(1), out.size(2))
        out = self.linear(out)
        return out, (h, c)