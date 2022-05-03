import torch
from transformers import BertJapaneseTokenizer, BertModel

model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)

tokens = tokenizer.tokenize("明日は自然言語処理の勉強をしよう。")
print(tokens)
