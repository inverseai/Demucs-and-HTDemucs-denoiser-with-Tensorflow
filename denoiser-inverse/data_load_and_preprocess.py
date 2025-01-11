import tensorflow as tf
import os
from denoiser.augment import augment


def load_audio(file_path, sample_rate):
    audio = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(contents=audio, desired_channels=1, desired_samples=sample_rate)
    audio = tf.squeeze(audio, axis=-1)  # Remove channel dimension
    return audio

def preprocess(noisy_path, clean_path, sample_rate, duration, split):
    noisy_audio = load_audio(noisy_path, sample_rate * duration)
    clean_audio = load_audio(clean_path, sample_rate * duration)

    # Ensure the audio length is consistent with the desired duration
    noisy_audio = noisy_audio[:sample_rate * duration]
    clean_audio = clean_audio[:sample_rate * duration]

    noisy_audio, clean_audio = tf.expand_dims(noisy_audio, axis=-1), tf.expand_dims(clean_audio, axis=-1)
    return noisy_audio, clean_audio

def get_dataset_from_directory(audio_dir, sample_rate, duration, split='train', batch_size=64, AUTOTUNE=tf.data.AUTOTUNE):
    noisy_dir = os.path.join(audio_dir, 'noisy')
    clean_dir = os.path.join(audio_dir, 'clean')
    
    noisy_files = sorted([os.path.join(noisy_dir, fname) for fname in os.listdir(noisy_dir)])
    clean_files = sorted([os.path.join(clean_dir, fname) for fname in os.listdir(clean_dir)])

    dataset = tf.data.Dataset.from_tensor_slices((noisy_files, clean_files))
    dataset = dataset.map(lambda noisy_path, clean_path: 
                          preprocess(noisy_path, clean_path, sample_rate, duration, split), 
                          num_parallel_calls=AUTOTUNE)
    if split == 'train':
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset


def get_dataset_train_raw(config, BATCH_SIZE):
    return get_dataset_from_directory(
        audio_dir=config['train_dir'],
        sample_rate=config['sample_rate'],
        duration=config['total_seconds_of_audio'],
        batch_size=BATCH_SIZE,
        AUTOTUNE=(tf.data.AUTOTUNE if config['autotune'] == -1 else config['autotune'])
    )

def get_dataset_val_raw(config, BATCH_SIZE):
    return get_dataset_from_directory(
        audio_dir=config['val_dir'],
        sample_rate=config['sample_rate'],
        duration=config['total_seconds_of_audio'],
        split='validate',
        batch_size=BATCH_SIZE,
        AUTOTUNE=(tf.data.AUTOTUNE if config['autotune'] == -1 else config['autotune'])
    )
