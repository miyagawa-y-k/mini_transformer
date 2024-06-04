import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils

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

#1モジュールのhead
class Head(nn.Module):
    #ヘッドの初期化
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        #マスク用の下三角行列
        self.register_buffer = ('tril', torch.tril(torch.ones(block_size, block_size)))
        #DropOut
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #input  (B, T, C) = (batch, time_step, channel)
        #output (batch, time_step, head_size)
        B, T, C = x.shape()
        k = self.key(x)     #(x)を(B, T, hs)にして返す
        q = self.query(x)   #(x)を(B, T, hs)にして返す

        #attentionの計算
        #qとkで内積をとって、ｋの次元で抑える
        #(B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * k.shape(-1)**0.5 
        #今読んでいる文字のところまで以外を、下三角行列でマスクする
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        #softmaxをかける
        wei = F.softmax(wei, dim=1)
        
        v = self.value(x)
        #attention_maskとvalueをかけて、出力を求める
        out = wei @ v
        return out

#headの並列化
class MultiHeadAttention(nn.Module):
    def __init__(self, num_head, head_size):
        super.__init__()
        #headを指定した数だけ並べる
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_head)])
        #線形写像をとる([連結したそれぞれの系列, 埋め込み次元])
        self.proj = nn.Linear(head_size * num_head, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #concat
        out = torch.cat([h(x) for h in self.heads], dim=1)
        #projection
        out = self.dropout(self.proj(out))
        return out

#FeedForward層の作成
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super.__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

#MultiHeadAttentionのBlock化
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super.__init__()
        head_size = n_embd // n_head
        #ScaledDotProductAttention
        self.sa = MultiHeadAttention(n_head, head_size)
        #FeedForward
        self.ffwd = FeedForward(n_embd)
        #saに対する残差接続
        self.ln1 = nn.LayerNorm(n_embd)
        #ffwdに対する残差接続
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


#
@torch.no_grad()
def estimate_loss():
    out = {}
    #model.eval()


    