import os
import yaml
import tensorflow as tf
import numpy as np
from denoiser.demucs import Demucs
from denoiser.htdemucs import HTDemucs
from denoiser import load
from tqdm import tqdm

def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def get_model(model_name, config):
    if model_name == 'demucs':
        model = Demucs(input_shape=(160000, 1))
    elif model_name == 'htdemucs':
        model = HTDemucs(**load.config)
        model.build(input_shape=(config['batch_size'],config['sample_rate']*config['total_seconds_of_audio'],config['channel']))
  
    return model

def load_checkpoint(checkpoint_dir, custom_objects=None):
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print("Loading model from checkpoint:", latest_checkpoint)
        model = tf.keras.models.load_model(latest_checkpoint, custom_objects=custom_objects)
        return model
    else:
        raise FileNotFoundError("No checkpoint found in directory: {}".format(checkpoint_dir))

def preprocess_audio(audio_file, sample_rate, duration):
    audio = tf.audio.decode_wav(tf.io.read_file(audio_file), desired_channels=1).audio
    audio = tf.expand_dims(audio, axis=0)  # Adding batch dimension
    return audio

def infer(audio_file, model, sample_rate=16000, duration=4):
    processed_audio = preprocess_audio(audio_file, sample_rate, duration)
    denoised_audio = model(processed_audio, training=False)
    denoised_audio = tf.squeeze(denoised_audio, axis=0)
    return denoised_audio

def save_audio(audio_tensor, output_path, sample_rate):
    audio_data = tf.audio.encode_wav(audio_tensor, sample_rate=sample_rate)
    tf.io.write_file(output_path, audio_data)
    print(f'Saved denoised audio to {output_path}')

def process_directory(input_dir, output_dir, model, sample_rate, duration):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith('.wav'):
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, filename)

            denoised_audio = infer(input_file, model, sample_rate, duration)
            save_audio(denoised_audio, output_file, sample_rate)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='config.yaml', help='Path to the config file')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the config file')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the input directory with noisy audio files')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save the denoised audio files')
    args = parser.parse_args()

    config = load_config(args.config_file)
    
    model = tf.keras.models.load_model(args.checkpoint_path)

    process_directory(args.input_dir, args.output_dir, model, sample_rate=config['sample_rate'], duration=config['total_seconds_of_audio'])
