from gc import callbacks
# from turtle import update
import tensorflow as tf
from demucs import Demucs
import sys
sys.path.append('..')
#from Diffq import DiffQuantiazer
#from stft_loss import custom_loss, loss_func, l1_mrstft_loss
import argparse
# import scipy.io.wavfile as wavfile
import re
import pickle
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, dest='model_path',
                        default= '/home/reyaj/Desktop/Diffq_Noise_reducer/Data/checkpoint_epoch-027_loss-0.434664.h5',
                        help='path of the weighted model'
                        '(default: %(default)s)')

    parser.add_argument('--noisy_path', type=str, dest='noisy_path',
                        default= '/home/reyaj/Desktop/Diffq_Noise_reducer/Data/jahaj_48.wav',
                        help='path of the noisy sample'
                        '(default: %(default)s)')
    parser.add_argument('--enhanced_path', type=str, dest='enhanced_path',
                        default= '/home/reyaj/Desktop/Diffq_Noise_reducer/Data/en_m.wav',
                        help='path of the enhanced sample'
                        '(default: %(default)s)')
    parser.add_argument('--batch_size', type=int, dest='batch_size',
                        default= 1,
                        help='the size of each batchs'
                        '(default: %(default)s)')
    return parser.parse_args()


class Enhancer:
    def __init__(self, model_path, noisy_path, enhanced_path, batch_size):
        self.model_path = model_path
        self.noisy_path = noisy_path
        self.enhanced_path = enhanced_path
        self.batch_size = batch_size

    def upd(self, batch, logs):
        self.progress = (batch + 1) / self.total_batch * 100
        print(self.progress)


    def pred(self, model = None):
        self_sample_rate = 16_000
        total_seconds_of_audio = 10
        total_number_of_sample = self_sample_rate * total_seconds_of_audio
        noisy_files = os.listdir(self.noisy_path)
        print(noisy_files)
        for noisy_path in noisy_files:
            print(noisy_path)
            raw_audio_noisy = tf.io.read_file(self.noisy_path + noisy_path)
            noisy, sample_rate = tf.audio.decode_wav( raw_audio_noisy, desired_channels=1)
            number_of_total_sample = noisy.shape[0]
            number_of_chunks = (number_of_total_sample + total_number_of_sample - 1) // total_number_of_sample
            if isinstance(model, type(None)):
                model = Demucs(input_shape=(total_number_of_sample, 1))
                model.load_weights(self.model_path)
            
            # for layer in model.layers:
            #      for param in layer.trainable_weights:
            #         print(param)

                # model = tf.keras.models.load_model(model_path, compile=False)
            number_of_total_sample_upperbound = number_of_chunks * total_number_of_sample
            padded_noisy = tf.pad(noisy, tf.constant([[0, number_of_total_sample_upperbound - number_of_total_sample], [0, 0]]), "CONSTANT")
            padded_noisy = tf.reshape(padded_noisy, [number_of_total_sample_upperbound // total_number_of_sample, total_number_of_sample, 1])
            self.total_batch = number_of_total_sample_upperbound // total_number_of_sample
            # print(self.total_batch)
            self.total_batch = (self.total_batch + self.batch_size - 1) // self.batch_size 
            lamda_call = tf.keras.callbacks.LambdaCallback(on_predict_batch_end=self.upd)
            estimate = model.predict(padded_noisy, batch_size=self.batch_size, callbacks=[lamda_call])
            estimate = tf.reshape(estimate, [number_of_total_sample_upperbound, 1])
            estimate = estimate[:number_of_total_sample,:]
            # dry= 0.1
            # estimate = (1 - dry) * estimate + dry * noisy
            file_name = os.path.splitext(os.path.basename(noisy_path))[0]
        
            tf.io.write_file(self.enhanced_path + file_name + ".wav", tf.audio.encode_wav(estimate, 16000))#))
            



def main(args):
    enhancer = Enhancer(args.model_path, args.noisy_path, args.enhanced_path, args.batch_size)
    enhancer.pred()
    # print(enhancer.progress)

if __name__ == '__main__':
    main(parse_args())


# python3 enhance_with_model.py --model_path  /home/ml-dev/Noise_reducer/out/checkpoint_25_nov/checkpoint_epoch-018_loss-0.002110.h5 --noisy_path /home/ml-dev/Data/Testing_noise_reducer/user_data/input_wav/ --enhanced_path /home/ml-dev/Data/Testing_noise_reducer/user_data/output_2110_dry_0.1/