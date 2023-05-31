from moviepy.editor import VideoFileClip
import pandas as pd


def extract_audio_from_video(data):
    for index, row in data.iterrows():
        dataset = row["Dataset"]
        path = row["AudioPath"]

        if dataset != "IEMOCAP":  # iemocap already in audio format
            index = path.rfind("mp4")
            audio_path = path[:index] + "wav"
            video = VideoFileClip(path)
            audio = video.audio
            audio.write_audiofile(audio_path)
            data.at[index, "AudioPath"] = audio_path
            return data
    return data


train_data = pd.read_csv("../datasets/Detoxy-B/train_cleaned.csv", encoding="latin-1")
train_data = extract_audio_from_video(train_data)
train_data.to_csv("datasets/Detoxy-B/train_cleaned.csv", index=False)

test_data = pd.read_csv("../datasets/Detoxy-B/test_cleaned.csv", encoding="latin-1")
test_data = extract_audio_from_video(test_data)
test_data.to_csv("datasets/Detoxy-B/test_cleaned.csv", index=False)
