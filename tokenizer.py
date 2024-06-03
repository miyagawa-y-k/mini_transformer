import torch
import torch.nn as nn
from torch.nn import functional as F

class tokenizer:
    def set_list(text):
        #受け取ったテキストをsetに入れてすべての文字でソートする
        chars = sorted(list(set(text)))
        #文字とそれに対応したidを相互に変換する辞書の作成
        char2int = {ch : i for i, ch in enumerate(chars)}
        int2char = {i : ch for i, ch in enumerate(chars)}

        return char2int, int2char

    def encoder(text, char2tint):
        #encoder：文字列を数値列に変換
        return lambda text : [char2tint[b] for b in text]

    def decoder(series, int2char):
        #受け取った系列を文字に戻す
        return lambda series : ''.join([int2char[b] for b in series])

    def echo():
        print('hello world')