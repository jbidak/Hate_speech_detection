import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

df_train = pd.read_csv("../datasets/Detoxy-B/train.csv", encoding="latin-1")
df_test = pd.read_csv("../datasets/Detoxy-B/test.csv", encoding="latin-1")

# ініціалізація лемматизатора та стоп-слів
nltk.download("wordnet")
nltk.download("stopwords")
nltk.download("punkt")
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def preprocess_text(text):
    text = re.sub(r"[^\w\s]", "", text)
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(tokens)


df_train = df_train[
    df_train["Dataset"].isin(["IEMOCAP", "MELD", "Common Voice", "CMU-MOSEI"])
]
df_test = df_test[
    df_test["Dataset"].isin(["IEMOCAP", "MELD", "Common Voice", "CMU-MOSEI"])
]

df_train["text"] = df_train["text"].apply(preprocess_text)
df_test["text"] = df_test["text"].apply(preprocess_text)

df_train.to_csv("datasets/Detoxy-B/train_cleaned.csv", index=False)
df_test.to_csv("datasets/Detoxy-B/test_cleaned.csv", index=False)
