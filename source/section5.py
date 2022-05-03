import torch
from transformers import BertForMaskedLM, BertJapaneseTokenizer
import numpy as np

model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
bert_mlm = BertForMaskedLM.from_pretrained(model_name)
bert_mlm = bert_mlm.cuda()

text = "今日は[MASK]へ行く。"

tokens = tokenizer.tokenize(text)
print(tokens)

input_ids = tokenizer.encode(text, return_tensors="pt")
input_ids = input_ids.cuda()

with torch.no_grad():
    output = bert_mlm(input_ids=input_ids)
    scores = output.logits

    mask_position = input_ids[0].tolist().index(4)

    id_best = scores[0, mask_position].argmax(-1).item()
    token_best = tokenizer.convert_ids_to_tokens(id_best)
    token_best = token_best.replace("##", "")

    text = text.replace("[MASK]", token_best)
    print(text)


print()
print("5-7")


def predict_mask_topk(text, tokenizer, bert_mlm, num_topk):
    input_ids = tokenizer.encode(text, return_tensors="pt")
    input_ids = input_ids.cuda()
    with torch.no_grad():
        output = bert_mlm(input_ids=input_ids)
    scores = output.logits

    mask_position = input_ids[0].tolist().index(4)
    topk = scores[0, mask_position].topk(num_topk)
    ids_topk = topk.indices
    tokens_topk = tokenizer.convert_ids_to_tokens(ids_topk)
    scores_topk = topk.values.cpu().numpy()
    text_topk = []
    for token in tokens_topk:
        token = token.replace("##", "")
        text_topk.append(text.replace("[MASK]", token, 1))

    return text_topk, scores_topk


text = "今日は[MASK]へ行く。"

text_topk, _ = predict_mask_topk(text, tokenizer, bert_mlm, 10)

print(*text_topk, sep="\n")

print()
print("5-8")


def greedy_prediction(text, tokenizer, bert_mlm):
    for _ in range(text.count("[MASK]")):
        text = predict_mask_topk(text, tokenizer, bert_mlm, 1)[0][0]
    return text


text = "今日は[MASK][MASK]へ行く。"
out = greedy_prediction(text, tokenizer, bert_mlm)
print(out)


print()
print("5-9")
text = "今日は[MASK][MASK][MASK][MASK][MASK]"
out = greedy_prediction(text, tokenizer, bert_mlm)
print(out)


print()
print("5-10")

def beam_search(text, tokenizer, bert_mlm, num_topk):
    num_mask = text.count("[MASK]")
    text_topk = [text]
    scores_topk = np.array([0])

    for _ in range(num_mask):
        text_candidates = []
        score_candidates = []

        for text_mask, score in zip(text_topk, scores_topk):
            text_topk_inner, scores_topk_inner = predict_mask_topk(
                text_mask, tokenizer, bert_mlm, num_topk
            )
            text_candidates.extend(text_topk_inner)
            score_candidates.append(score + scores_topk_inner)
        
        score_candidates = np.hstack(score_candidates)
        idx_list = score_candidates.argsort()[::-1][:num_topk]
        text_topk = [text_candidates[idx] for idx in idx_list]
        scores_topk = score_candidates[idx_list]
    
    return text_topk

text = "今日は[MASK][MASK]へ行く。"
text_topk = beam_search(text, tokenizer, bert_mlm, 10)
print(*text_topk, sep="\n")


print()
print("5-11")
text = "今日は[MASK][MASK][MASK][MASK][MASK]"
text_topk = beam_search(text, tokenizer, bert_mlm, 10)
print(*text_topk, sep="\n")
