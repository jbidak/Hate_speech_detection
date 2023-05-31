import pickle
import torch
from transformers import BertTokenizer, BertModel
import pandas as pd

max_seq_length = 512
device = torch.device("cpu")
model = BertModel.from_pretrained("bert-base-uncased").to(device)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def text_to_embeddings(data):
    bert_embeddings = []
    for text in data:
        text = str(text)[:max_seq_length]
        tokens = tokenizer.encode(text, add_special_tokens=True)
        tokens_tensor = torch.tensor([tokens]).to(device)
        with torch.no_grad():
            print(model(tokens_tensor))
            res = model(tokens_tensor).last_hidden_state
        bert_embeddings.append(res)
    return bert_embeddings


train_data = pd.read_csv("dataset/Detoxy-B/train_cleaned.csv", encoding="latin-1")
train_bert_embeddings = text_to_embeddings(train_data["text"])
f_train = open("train_bert_embeddings.bin", "wb")
pickle.dump(train_bert_embeddings, f_train)
f_train.close()

test_data = pd.read_csv("dataset/Detoxy-B/test_cleaned.csv", encoding="latin-1")
test_bert_embeddings = text_to_embeddings(test_data["text"])
f_test = open("test_bert_embeddings.bin", "wb")
pickle.dump(test_bert_embeddings, f_test)
f_test.close()
