import pandas as pd


def create_audio_path(data):
    data["AudioPath"] = ""
    for index, row in data.iterrows():
        dataset = row["Dataset"]
        filename = row["FileName"]

        if dataset == "CMU-MOSEI":
            modified_filename = filename.rsplit("_", 1)
            modified_filename[-1] = modified_filename[-1].replace("_", "/")
            audio_path = "datasets/CMU-MOSEI/" + "/".join(modified_filename) + ".mp4"
            data.at[index, "AudioPath"] = audio_path
        elif dataset == "IEMOCAP":
            audio_path = "datasets/IEMOCAP" + "/" + filename + ".wav"
            data.at[index, "AudioPath"] = audio_path
        elif dataset == "MELD":
            parts = filename.split("_")
            audio_path = "datasets/MELD/" + f"{parts[0]}/{parts[1]}_{parts[2]}.mp4"
            data.at[index, "AudioPath"] = audio_path
        elif dataset == "Common Voice":
            audio_path = "datasets/COMMON-VOICE/" + filename + ".mp4"
            data.at[index, "AudioPath"] = audio_path
    return data


train_data = pd.read_csv("../datasets/Detoxy-B/train_cleaned.csv", encoding="latin-1")
train_data = create_audio_path(train_data)
train_data.to_csv("datasets/Detoxy-B/train_cleaned.csv", index=False)

test_data = pd.read_csv("../datasets/Detoxy-B/test_cleaned.csv", encoding="latin-1")
test_data = create_audio_path(test_data)
test_data.to_csv("datasets/Detoxy-B/test_cleaned.csv", index=False)
