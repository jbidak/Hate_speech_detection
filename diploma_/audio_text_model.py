import pickle

import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import class_weight

from keras.layers import Dense, Dropout, concatenate
from keras.models import Model
from keras import Input


DATASET_DIR = "../datasets/Detoxy-B/"

TEXT_LEN = 768
AUDIO_LEN = 1024


def create_nn_merge_model():
    audio_input = Input(shape=(1024,))
    text_input = Input(shape=(768,))

    dropout_audio = Dropout(0.3)(audio_input)
    audio_layer = Dense(100, activation="tanh")(dropout_audio)

    dropout_audio_layer = Dropout(0.3)(audio_layer)

    merged_inputs = concatenate([dropout_audio_layer, text_input])

    dense_layer1 = Dense(TEXT_LEN + 100, activation="tanh")(merged_inputs)
    dense_layer2 = Dense(100, activation="tanh")(dense_layer1)
    dense_layer3 = Dense(50, activation="tanh")(dense_layer2)

    output_layer = Dense(1, activation="sigmoid")(dense_layer3)

    model = Model(inputs=[audio_input, text_input], outputs=output_layer)
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        metrics=["accuracy"],
    )

    return model


def get_results(model, X, Y):
    y_pred = model.predict(X)
    y_pred = np.rint(y_pred)
    print(f"Accuracy: {round(accuracy_score(Y, y_pred), 2)}")
    print(f"Precision: {round(precision_score(Y, y_pred), 2)}")
    print(f"Recall: {round(recall_score(Y, y_pred), 2)}")
    print(f"F1_score: {round(f1_score(Y, y_pred), 2)}\n\n")


def train_and_predict(df_train, df_test):
    X_train = np.array(df_train["speech"].tolist())
    berts_train = np.array(df_train["bert"].tolist())
    Y_train = np.array(df_train["label2a"].tolist())

    X_test = np.array(df_test["speech"].tolist())
    berts_test = np.array(df_test["bert"].tolist())
    Y_test = np.array(df_test["label2a"].tolist())

    model = create_nn_merge_model()
    model.reset_states()
    weights = class_weight.compute_class_weight(
        "balanced", classes=np.unique(df_train["label2a"]), y=df_train["label2a"]
    )
    weights = {i: weights[i] for i in range(len(np.unique(df_train["label2a"])))}

    model.fit(
        [X_train, berts_train],
        Y_train,
        epochs=100,
        class_weight=weights,
        validation_data=([X_test, berts_test], Y_test),
    )
    print("Performance on test set:")
    get_results(model, [X_test, berts_test], Y_test)


def main():
    df_train = pd.read_csv(DATASET_DIR + "train_vec.csv")
    df_test = pd.read_csv(DATASET_DIR + "test_vec.csv")
    train_and_predict(df_train, df_test)


if __name__ == "__main__":
    main()
