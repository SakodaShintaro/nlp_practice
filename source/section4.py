import torch
from transformers import BertJapaneseTokenizer, BertModel

model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)

tokens = tokenizer.tokenize("明日は自然言語処理の勉強をしよう。")
print(tokens)

tokens = tokenizer.tokenize("明日はマシンラーニングの勉強をしよう。")
print(tokens)

tokens = tokenizer.tokenize("機械学習を中国語にすると机器学习だ。")
print(tokens)

input_ids = tokenizer.encode("明日は自然言語処理の勉強をしよう。")
print(input_ids)

print(tokenizer.convert_ids_to_tokens(input_ids))

text = "明日の天気は晴れだ。"
encoding = tokenizer(
    text, max_length=12, padding="max_length", truncation=True
)
print("# encoding:", encoding)

tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
print("# tokens:", tokens)

# max_lengthを6にする場合
encoding = tokenizer(
    text, max_length=6, padding="max_length", truncation=True
)
print("# encoding:", encoding)

tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
print("# tokens:", tokens)
print()

print("複数文の入力")
text_list = ["明日の天気は晴れだ。", "パソコンが急に動かなくなった。"]
print(tokenizer(text_list, max_length=10, padding="max_length", truncation=True))
print()

print("padding=longestの利用")
print(tokenizer(text_list, padding="longest", truncation=True))
print()

print("return_tensors=pt")
print(tokenizer(text_list, padding="longest", truncation=True, return_tensors="pt"))
print()

print("BERTモデルの利用")
bert = BertModel.from_pretrained(model_name)
bert = bert.cuda()

print(bert.config)

text_list = [
    "明日は自然言語処理の勉強をしよう。",
    "明日はマシンラーニングの勉強をしよう。"
]

encoding = tokenizer(
    text_list,
    max_length=32,
    padding="max_length",
    truncation=True,
    return_tensors="pt"
)

encoding = {k: v.cuda() for k, v in encoding.items()}

output = bert(**encoding)
last_hidden_state = output.last_hidden_state

print(last_hidden_state)

print(last_hidden_state.size())
