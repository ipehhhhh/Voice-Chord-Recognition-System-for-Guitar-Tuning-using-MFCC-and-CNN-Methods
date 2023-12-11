import librosa
import os
import json
import matplotlib.pyplot as plt
import numpy as np

num_mfcc = 13 
n_fft = 2048
hop_length = 512

# Visual MFCC
def visualize_mfcc(audio_files):
   for i, (class_label, file_path) in enumerate(audio_files.items(), start=1):
       # Load the original signal
       signal, sr = librosa.load(file_path)
       
       # Plot the original signal
       plt.figure(i*2-1)
       time = np.arange(0, len(signal)) / sr
       plt.plot(time, signal)
       plt.title(f'Class: {class_label} - Original Signal')
       plt.xlabel('Time (s)')
       plt.ylabel('Amplitude')
       plt.show()

       # Plot the MFCCs
       plt.figure(i*2)
       mfccs = librosa.feature.mfcc(y=signal, n_mfcc=13, sr=sr)
       librosa.display.specshow(mfccs, x_axis="time", sr=sr)
       plt.colorbar(format="%+2.f")
       plt.title(f'Class: {class_label} - MFCCs')
       plt.show()


if __name__ == "__main__":
    audio_files = {
        "Am": r"C:\Users\syari\Document\KULIAH\Semester 5\Pemrosesan suara\extractedwav\Am\Am_acousticguitar_Mari_1.wav",
        "Bb": r"C:\Users\syari\Document\KULIAH\Semester 5\Pemrosesan suara\extractedwav\Bb\Bb_acousticguitar_Mari_1.wav",
        "Bdim": r"C:\Users\syari\Document\KULIAH\Semester 5\Pemrosesan suara\extractedwav\Bdim\Bdim_acousticguitar_Mari_1.wav",
        "C": r"C:\Users\syari\Document\KULIAH\Semester 5\Pemrosesan suara\extractedwav\C\C_acousticguitar_Mari_1.wav",
        "Dm": r"C:\Users\syari\Document\KULIAH\Semester 5\Pemrosesan suara\extractedwav\Dm\Dm_acousticguitar_Mari_1.wav",
        "Em": r"C:\Users\syari\Document\KULIAH\Semester 5\Pemrosesan suara\extractedwav\Em\Em_acousticguitar_Mari_1.wav",
        "F": r"C:\Users\syari\Document\KULIAH\Semester 5\Pemrosesan suara\extractedwav\F\F_acousticguitar_Mari_1.wav",
        "G": r"C:\Users\syari\Document\KULIAH\Semester 5\Pemrosesan suara\extractedwav\G\G_acousticguitar_Mari_1 17.51.39.wav"
    }

    visualize_mfcc(audio_files)

# Define your dataset path and output JSON path
DATASET_PATH = r"C:\Users\syari\Document\KULIAH\Semester 5\Pemrosesan suara\extractedwav"
JSON_PATH = r"hasil.json"
SAMPLES_TO_CONSIDER = 22050  # 1 sec. of audio

def preprocess_dataset(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512):
    data = {
        "mapping": [],
        "labels": [],
        "MFCCs": [],
        "files": []
    }

    # Create a dictionary to map folder names to labels
    label_mapping = {}

    # Get a list of subdirectories in the dataset_path
    subdirs = os.listdir(dataset_path)

    for i, subdir in enumerate(subdirs):
        label_mapping[subdir] = i

    for subdir in subdirs:
        subdir_path = os.path.join(dataset_path, subdir)

        if os.path.isdir(subdir_path):
            data["mapping"].append(subdir)
            label = label_mapping[subdir]

            for filename in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, filename)

                # Load audio file and slice it to ensure length consistency among different files
                signal, sample_rate = librosa.load(file_path)

                if len(signal) >= SAMPLES_TO_CONSIDER:
                    signal = signal[:SAMPLES_TO_CONSIDER]

                    # Extract MFCCs
                    MFCCs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13, n_fft=2048, hop_length=512)

                    # Store data for analyzed track
                    data["MFCCs"].append(MFCCs.T.tolist())
                    data["labels"].append(label)
                    data["files"].append(file_path)

    # Save data in a JSON file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

if __name__ == "__main__":
    preprocess_dataset(DATASET_PATH, JSON_PATH)

