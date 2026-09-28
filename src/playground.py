import librosa
import matplotlib.pyplot as plt
import os
import subprocess

import pandas as pd
df = pd.read_csv("../data/coughvid_20211012/metadata_compiled.csv")
print(df.shape)        # how many rows/columns total
print(df.columns)      # see every column name that exists
print(df["status"].value_counts())   # see how many of each label exist

# audio_file, sample_rate = librosa.load("",sr=None)


# plt.plot(audio_file)
# plt.title("Raw Waveform")
# plt.show()




# create a healthy original dataframe

healthy_df = df[(df['status'] == 'healthy') & (df['cough_detected'] >= 0.8)].copy()
symptomatic_covid_df = df[(df['status'].isin(['symptomatic', 'COVID-19'])) & (df['cough_detected'] >= 0.8)].copy()
print("Filtered Dataframe: ")
print(healthy_df.head(5))
print("--------------------")
print("Ill Dataframe")
print(symptomatic_covid_df.head(5))

# Now we need to take a sample of each
healthy_sample_df = healthy_df.sample(n=200, random_state = 42)

sick_sample_df = symptomatic_covid_df.sample(n=200, random_state = 42)

# Add label to simplify labels - 0 is healthy, 1 is not
healthy_sample_df['label'] = 0
sick_sample_df['label'] = 1

# Pull the uuid column into a plain python list using .toList()

healthy_uuid = healthy_sample_df['uuid'].tolist()
print(healthy_uuid)


sick_uuid = sick_sample_df['uuid'].tolist()

# Concatenate into one giant dataframe for training and testing purposes

table = pd.concat([healthy_sample_df, sick_sample_df]).sample(frac=1, random_state=42)

print(table.head(5))



# Now that we have the patients we are going to work with, we will need to translate their oog to wav files, now it would be easier to do

# Create the output folders for each

healthy_audio = "../data/healthy"
os.makedirs(healthy_audio, exist_ok=True)

sick_audio = "../data/sick"
os.makedirs(sick_audio, exist_ok=True)


# Extract array of cols

array_uuid_label = table[['uuid', 'label']].to_numpy()
# print(array_uuid_label)
# for each instance, built the input file path: raw_folder (data) + uuid + ".oog | webm"
for row in array_uuid_label:
    input_file_path = "../data/coughvid20211012" + row[0]

    if row[1] is 0:
        output_file = "../data/healthy" + row[0] + ".wav"
    else:
        output_file = "../data/sick" + row[0] + ".wav"

    # run the script, cover the script with a try and catch clause
    try{
        input_file_path+=".oog"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", input_path, output_path], check=True)
    }
    except FileNotFoundError:
        input_file_path+=".webm"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", input_path, output_path], check=True)


