import pickle
import torch
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from torch.nn.utils.rnn import pad_sequence


df_train = pd.read_csv("dataset/Detoxy-B/train_cleaned.csv")
df_test = pd.read_csv("dataset/Detoxy-B/test_cleaned.csv")

f_train = open("train_bert_embeddings.bin", "rb")
train_embeddings = pickle.load(f_train)
f_train.close()

f_test = open("test_bert_embeddings.bin", "rb")
test_embeddings = pickle.load(f_test)
f_test.close()

train_labels = df_train["label2a"]
test_labels = df_test["label2a"]


train_features = pad_sequence(
    [torch.flatten(embedding) for embedding in train_embeddings], batch_first=True
)
test_features = pad_sequence(
    [torch.flatten(embedding) for embedding in test_embeddings], batch_first=True
)

train_features = train_features.detach().numpy()
test_features = test_features.detach().numpy()

train_labels = (
    train_labels.values if isinstance(train_labels, pd.Series) else train_labels
)
test_labels = test_labels.values if isinstance(test_labels, pd.Series) else test_labels

# ------------------------------------------------------------------------------
"""Логістична регресія"""
logreg = LogisticRegression(verbose=1)

print("Logistic regression train starts.")
# тренування логістичної регресії
logreg.fit(train_features, train_labels)

test_preds = logreg.predict(test_features)

# Оцінка результатів на тренувальному наборі
lr_accuracy = accuracy_score(test_labels, test_preds)
lr_precision = precision_score(test_labels, test_preds)
lr_recall = recall_score(test_labels, test_preds)
lr_f1_score = f1_score(test_labels, test_preds)

print("---Logistic regression on test data---.")
print(f"Accuracy: {lr_accuracy}")
print(f"Precision: {lr_precision}")
print(f"Recall: {lr_recall}")
print(f"F1-score: {lr_f1_score}")
print("-----------------------------")
print()
# -----------------------------------------------------------------------------
"""K-Nearest Neighbors"""
knn = KNeighborsClassifier()

# Тренування KNN на тренувальних даних
knn.fit(train_embeddings, train_labels)

# Прогнозування на валідаційному наборі
test_preds = knn.predict(test_embeddings)

# Оцінка результатів на валідаційному наборі
val_accuracy = accuracy_score(test_labels, test_preds)
val_precision = precision_score(test_labels, test_preds)
val_recall = recall_score(test_labels, test_preds)
val_f1_score = f1_score(test_labels, test_preds)

print(f"Accuracy on test set: {val_accuracy}")
print(f"Precision on test set: {val_precision}")
print(f"Recall on test set: {val_recall}")
print(f"F1-score on test set: {val_f1_score}")
# # # -----------------------------------------------------------------------------

# # -----------------------------------------------------------------------------
"""Support Vector Machine (SVM)"""
svm = SVC()

svm.fit(train_embeddings, train_labels)

test_preds = svm.predict(test_embeddings)

val_accuracy = accuracy_score(test_labels, test_preds)
val_precision = precision_score(test_labels, test_preds)
val_recall = recall_score(test_labels, test_preds)
val_f1_score = f1_score(test_labels, test_preds)

print(f"Accuracy on test set: {val_accuracy}")
print(f"Precision on test set: {val_precision}")
print(f"Recall on test set: {val_recall}")
print(f"F1-score on test set: {val_f1_score}")
# -----------------------------------------------------------------------------
