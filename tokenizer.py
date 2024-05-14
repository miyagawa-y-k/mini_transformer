import argparse

parser = argparse.ArgumentParser(description='コマンドライン引数で変数を初期化する例')

# コマンドライン引数を定義
parser.add_argument('--name', type=str, help='名前を入力してください', required=True)
parser.add_argument('--age', type=int, help='年齢を入力してください', required=True)
parser.add_argument('--height', type=float, help='身長を入力してください', required=False, default=170.0)

# 引数をパースして、argsオブジェクトに格納
args = parser.parse_args()

# 変数の初期化
name = args.name
age = args.age
height = args.height

with  open('./Documents/git_clones/mini_transformer/sample.txt', 'r' ,encoding='utf-8') as f:
    text = f.read()

# 変数の出力
print(f'名前: {name}')
print(f'年齢: {age}')
print(f'身長: {height}')
print(text)