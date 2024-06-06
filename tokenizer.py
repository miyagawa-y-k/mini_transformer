import torch
import torch.nn as nn
from torch.nn import functional as F

class Tokenizer():
    def __init__(self):
        self.chars = set()
        self.char2int = {}
        self.int2char = {}
        self.train_data = []
        self.val_data = []
        #固定値
        self.batch_size = 64
        self.block_size = 256
        self.max_iters = 5000
        self.eval_interval = 500
        self.learning_rate = 3e-4
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.eval_iters = 200
        self.n_embd = 384
        self.n_head = 6
        self.n_layer = 6
        self.dropout = 0.2

    def set_list(self, text):
        #受け取ったテキストをsetに入れてすべての文字でソートする
        self.chars = sorted(list(set(text)))
        #文字とそれに対応したidを相互に変換する辞書の作成
        self.char2int = {ch : i for i, ch in enumerate(self.chars)}
        self.int2char = {i : ch for i, ch in enumerate(self.chars)}

    def encoder(self, text):
        #encoder：文字列を数値列に変換
        return list(self.char2int[b] for b in text)

    def decoder(self, tokens):
        #受け取ったものがlist型ではない場合は変換
        if type(tokens) != list:
            tokens = tokens.tolist()
        #受け取った系列を文字に戻す
        return ''.join([self.int2char[b] for b in tokens])

    def train_test_split(self, text):
        #入力データを訓練データとテストデータに分割
        data = torch.tensor(self.encoder(text), dtype=torch.long)
        n = int(0.9 * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]
        return self.train_data, self.val_data

    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size, ))
        x = torch.stack([data[i: i+self.block_size] for i in ix])
        y = torch.stack([data[i+1: i+self.block_size+1] for i in ix])
        x, y = x.to(self.device), y.to(self.device)
        return x, y
    
    
    
    
    
    
    
    