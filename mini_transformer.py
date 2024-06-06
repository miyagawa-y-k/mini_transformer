import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils
import sys

#1モジュールのhead
class Head(nn.Module):
    #ヘッドの初期化
    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        #マスク用の下三角行列
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        #DropOut
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #input  (B, T, C) = (batch, time_step, channel)
        #output (batch, time_step, head_size)
        B, T, C = x.shape
        k = self.key(x)     #(x)を(B, T, hs)にして返す
        q = self.query(x)   #(x)を(B, T, hs)にして返す

        #attentionの計算
        #qとkで内積をとって、ｋの次元で抑える
        #(B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**0.5 
        #今読んでいる文字のところまで以外を、下三角行列でマスクする
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        #softmaxをかける
        wei = F.softmax(wei, dim=-1)
        
        v = self.value(x)
        #attention_maskとvalueをかけて、出力を求める
        out = wei @ v
        return out

#headの並列化
class MultiHeadAttention(nn.Module):
    def __init__(self, head_size, num_embd, block_size, num_head, dropout):
        super().__init__()
        #headを指定した数だけ並べる
        self.heads = nn.ModuleList([Head(head_size, num_embd, block_size, dropout) for _ in range(num_head)])
        #線形写像をとる([連結したそれぞれの系列, 埋め込み次元])
        self.proj = nn.Linear(head_size * num_head, num_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #concat
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        #projection
        out = self.dropout(self.proj(out))
        return out

#FeedForward層の作成
class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
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
    def __init__(self, n_embd, block_size, n_head, dropout):
        super().__init__()
        head_size = n_embd // n_head
        #ScaledDotProductAttention
        self.sa = MultiHeadAttention(head_size, n_embd, block_size, n_head, dropout)
        #FeedForward
        self.ffwd = FeedForward(n_embd, dropout)
        #saに対する残差接続
        self.ln1 = nn.LayerNorm(n_embd)
        #ffwdに対する残差接続
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, chars = None, *, batch_size=64, block_size=256, max_iters=5000, eval_interval=500, learning_rate=3e-4,device = 'cuda:0' if torch.cuda.is_available() else 'cpu', eval_iters=200, n_embd=384, n_head=6, n_layer=6, dropout=0.2):
        super().__init__()
        #例外処理
        if type(chars) != list:
            print('Input is list only.')
            sys.exit(1)
        elif len(chars) == 0:
            print('Please input a valid list.')
            sys.exit(1)
        else:
            pass
        
        #固定値
        self.batch_size = batch_size
        self.block_size = block_size
        self.max_iters = max_iters
        self.eval_interval = eval_interval
        self.learning_rate = learning_rate
        self.device = device
        self.eval_iters = eval_iters
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.vocab_size = len(chars)
        self.dropout = dropout

        #各モジュールの初期化
        self.token_embedding_table = nn.Embedding(self.vocab_size, self.n_embd)
        self.position_embedding_table = nn.Embedding(self.block_size, self.n_embd)
        self.blocks = nn.Sequential(*[Block(self.n_embd, self.block_size, self.n_head, self.dropout) for _ in range(self.n_layer)])
        self.ln_f = nn.LayerNorm(self.n_embd)
        self.ln_head = nn.Linear(self.n_embd, self.vocab_size)

        #_init_weightsを使って重みを初期化
        self.apply(self._init_weights)

    #初期化関数
    def _init_weights(self, module):
        #対象のモジュールが線形層のとき、平均=0.0、偏差=0.2で初期化
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.2)
            #対象のモジュールにバイアスがあるとき、0で初期化
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
         #対象のモジュールが埋め込み層のとき、平均=0.0、偏差=0.2で初期化
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.2)

    #順伝搬
    def forward(self, idx, targets=None):
        idx = idx.to(self.device)
        #
        if targets is not None:
            targets = targets.to(self.device)

        B, T = idx.shape

        #トークンの埋め込みと位置埋め込みをそれぞれ別に作成
        tok_emb = self.token_embedding_table(idx)   #(B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))    #(T, C)
        #二つの埋め込みをまとめる
        x = tok_emb + pos_emb   #(B, T, C)
        #順伝搬
        x = self.blocks(x)      #(B, T, C)
        x = self.ln_f(x)        #(B, T, C)
        logits = self.ln_head(x)    #(B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            #
            logits = logits.view(B*T, C)    #(B*T, C)
            #
            targets = targets.view(B*T)     #(B*T)
            #lossの計算
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_token):
        #
        for _ in range(max_new_token):
            idx = idx.to(self.device)
            #タイムステップまでで区切る
            idx_cond = idx[:, -self.block_size:]
            #予測
            logits, loss = self(idx_cond)
            #最後のタイムステップの部分だけ取り出す
            logits = logits[:, -1, :]   #(B, C)
            #確率の計算
            probs = F.softmax(logits, dim=-1)   #(B, C)
            #probに与えられた確率分布に基づいて、トークンを選択
            idx_next = torch.multinomial(probs, num_samples=1)  #(B, 1)
            #今処理しているシーケンスに、予測したトークンをつなげる
            idx = torch.cat((idx, idx_next), dim=1) #(B, T+1)
        return idx


    