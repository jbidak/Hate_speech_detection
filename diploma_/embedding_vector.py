import librosa
import numpy as np
import torch
import torchaudio
from pandas import read_csv
from transformers import Wav2Vec2FeatureExtractor, AutoConfig
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model,
)
from datasets import load_dataset

AUDIO_TEST_DF_PATH = "../datasets/Detoxy-B/test_cleaned.csv"
AUDIO_TRAIN_DF_PATH = "../datasets/Detoxy-B/train_cleaned.csv"

SAVE_DIR = "../datasets/Detoxy-B/"
PRETRAINED_SER_MODEL = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(PRETRAINED_SER_MODEL)


def load_data_frame(path):
    df = read_csv(path)
    print(df.head())
    return df


def export_data_frame(df, save_path):
    df.to_csv(save_path, sep="\t", encoding="utf-8", index=False)


def extract_and_export_vectors(df, name):
    result = df.map(get_vec_from_pre_trained, batched=True, batch_size=2)
    result.to_csv(f"{SAVE_DIR}/{name}_vecs.csv", sep=",", encoding="utf-8", index=False)


class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)

        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merged_strategy(
        self,
        hidden_states,
    ):
        return torch.mean(hidden_states, dim=1)

    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)

        return hidden_states


def get_vec_from_pre_trained(batch):
    device = torch.device("cpu")
    config = AutoConfig.from_pretrained(PRETRAINED_SER_MODEL)
    setattr(config, "pooling_mode", "mean")
    model = Wav2Vec2ForSpeechClassification.from_pretrained(
        PRETRAINED_SER_MODEL,
        config=config,
    ).to(device)

    features = feature_extractor(
        batch["speech"],
        sampling_rate=feature_extractor.sampling_rate,
        return_tensors="pt",
        padding=True,
    )
    input_values = features.input_values.to(device)
    with torch.no_grad():
        res = model(input_values)
        res = res.tolist()
    batch["vec"] = res
    print(res)
    return batch


def speech_file_to_array_fn2(batch):
    speech_array, sampling_rate = torchaudio.load(batch["AudioPath"])
    speech_array = speech_array.squeeze().numpy().flatten()
    speech_array = librosa.resample(
        np.asarray(speech_array), sampling_rate, feature_extractor.sampling_rate
    )
    batch["speech"] = speech_array
    return batch


def main():
    df_test = load_data_frame(AUDIO_TEST_DF_PATH)
    df_train = load_data_frame(AUDIO_TRAIN_DF_PATH)

    export_data_frame(df_test, f"{SAVE_DIR}/test_e.csv")
    export_data_frame(df_train, f"{SAVE_DIR}/train_e.csv")

    test_df = load_dataset(
        "csv", data_files={"test_df": "SAVE_DIR/test_e.csv"}, delimiter="\t"
    )["test_df"]
    train_df = load_dataset(
        "csv", data_files={"train_df": "SAVE_DIR/train_e.csv"}, delimiter="\t"
    )["train_df"]

    test_df = test_df.map(speech_file_to_array_fn2)
    train_df = train_df.map(speech_file_to_array_fn2)

    extract_and_export_vectors(test_df, "test")
    extract_and_export_vectors(train_df, "train")


if __name__ == "__main__":
    main()
