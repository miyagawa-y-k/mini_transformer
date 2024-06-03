import torch
import torch.nn as nn
from torch.nn import functional as F

#固定値
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

def set_list(text):
    #受け取ったテキストをsetに入れてすべての文字でソートする
    chars = sorted(list(set(text)))
    #文字とそれに対応したidを相互に変換する辞書の作成
    char2int = {ch : i for i, ch in enumerate(chars)}
    int2char = {i : ch for i, ch in enumerate(chars)}
    return char2int, int2char

def encoder(text, char2tint):
    #encoder：文字列を数値列に変換
    return list(char2tint[b] for b in text)

def decoder(tokens, int2char):
    #受け取ったものがlist型ではない場合は変換
    if type(tokens) != list:
        tokens = tokens.tolist()
    #受け取った系列を文字に戻す
    return ''.join([int2char[b] for b in tokens])

def train_test_split(text, char2int):
    #入力データを訓練データとテストデータに分割
    data = torch.tensor(encoder(text, char2int), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    valid_data = data[n:]
    return train_data, valid_data

def get_batch(divided_data):
    ix = torch.randint(len(divided_data) - block_size, (batch_size, ))
    x = torch.stack([divided_data[i: i+block_size] for i in ix])
    y = torch.stack([divided_data[i+1: i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
    
    
    
    
    
    
    
    