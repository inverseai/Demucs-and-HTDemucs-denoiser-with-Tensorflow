import os
import numpy as np
import librosa
import tensorflow as tf

from htdemucs import HTDemucs
import load

def load_audio_files_from_folder(folder_path):
    audio_data = []
    file_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.wav')]
    for file_path in file_list:
        audio, _ = librosa.load(file_path, sr=16000)
        # print(len(audio))
        audio_data.append(audio)
    return audio_data


# Load audio files
clean_folder = "/mnt/4tb/Saiham/train_dataset_2nd/sample_clean"
noisy_folder = "/mnt/4tb/Saiham/train_dataset_2nd/sample_noisy"

clean_audio = load_audio_files_from_folder(clean_folder)
noisy_audio = load_audio_files_from_folder(noisy_folder)

# Pad sequences
clean_features = tf.keras.preprocessing.sequence.pad_sequences(clean_audio, dtype='float32', padding='post')
noisy_features = tf.keras.preprocessing.sequence.pad_sequences(noisy_audio, dtype='float32', padding='post')

# Reshape to match the model's expected input
clean_features = clean_features[..., np.newaxis]  # Adds an extra dimension, making it (batch_size, time_steps, 13, 1)
noisy_features = noisy_features[..., np.newaxis]

# Load the pre-trained model
model = HTDemucs(**load.config)
model.build(input_shape=(1, 160000, 1))
model.load_weights("/mnt/4tb/Saiham/htd/checkpoints.weights.h5")

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

print("Training...")
# Fit the model
history = model.fit(
    x=noisy_features,
    y=clean_features,
    batch_size=4,
    validation_split=0.1,
    epochs=30  # Adjust as necessary
)

print("Training complete.")
print(history.history)