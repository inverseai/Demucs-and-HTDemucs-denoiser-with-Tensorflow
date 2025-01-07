import os
import tensorflow as tf

from denoiser.demucs import Demucs
from denoiser.stft_loss import custom_loss, loss_func, l1_mrstft_loss
import argparse
import re
from scipy import signal
import numpy as np
from pystoi import stoi
from pesq import pesq

AUTOTUNE = tf.data.experimental.AUTOTUNE


# Required packages to install
# pip3 install pystoi
# pip3 install pesq

#sources : https://github.com/ludlows/python-pesq
#sources : https://github.com/mpariente/pystoi

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, dest='model_path',
                        default= 'model',
                        help='path of the weighted model'
                        '(default: %(default)s)')

    parser.add_argument('--tfrecord_path', type=str, dest='tfrecord_path',
                        default= 'tfrecords',
                        help='path of the tfrecord sample'
                        '(default: %(default)s)')
    
    return parser.parse_args()

def _parse_batch(record_batch, sample_rate, duration):
    n_samples = sample_rate * duration

    # Create a description of the features
    feature_description = {
        'noisy': tf.io.FixedLenFeature([n_samples], tf.float32),
        'clean': tf.io.FixedLenFeature([n_samples], tf.float32),
    }

    # Parse the input `tf.Example` proto using the dictionary above
    example = tf.io.parse_example(record_batch, feature_description)

    return example['noisy'], example['clean']

def get_dataset_from_tfrecords(tfrecords_dir='tfrecords', split='test',
                               batch_size=1, sample_rate=16000, duration=10,
                               n_epochs=10):
    if split not in ('train', 'test', 'validate'):
        raise ValueError("split must be either 'train', 'test' or 'validate'")

    # List all *.tfrecord files for the selected split
    pattern = os.path.join(tfrecords_dir, '{}*.tfrecord'.format(split))
    files_ds = tf.data.Dataset.list_files(pattern)

    # Disregard data order in favor of reading speed
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    files_ds = files_ds.with_options(ignore_order)

    # Read TFRecord files in an interleaved order
    ds = tf.data.TFRecordDataset(files_ds,
                                 compression_type='ZLIB',
                                 num_parallel_reads=AUTOTUNE)
 #   ds = ds.shuffle(200, reshuffle_each_iteration=True)
    # Prepare batches
    ds = ds.batch(batch_size)

    # Parse a batch into a dataset of [audio, label] pairs
    ds = ds.map(lambda x: _parse_batch(x, sample_rate, duration))

    # Repeat the training data for n_epochs. Don't repeat test/validate splits.
    # if split == 'train':
    #     ds = ds.repeat(n_epochs)

    return ds.prefetch(buffer_size=AUTOTUNE)

def evaluate_stoi_pesq(train_ds,model):
    stoi_sum=0
    total_element = 0
    pesq_sum=0
    
    i = -1
    for sample in train_ds:
        i += 1
        #if i>2 : break
        x = sample[0][0]     #noisy
        y = sample[1][0]     #clean  

        
        y_p = tf.reshape(model.predict(tf.reshape(x, [1,-1,1])), [-1])
        
       # print("Y_ ",y_p.shape, x.shape)
        dry = 0.05
        y_p = (1 - dry) * y_p + dry * x 

        try:
            #Stoi Evaluation
            eval_score_stoi = stoi(y,y_p,16000,extended=False)
            #print("Test")

            #Pesq Evaluation
            ratio = float(16000) / 16000
            n_samples = int(np.ceil(y.shape[-1] * ratio))
            y = signal.resample(y, n_samples)
            y_p = signal.resample(y_p, n_samples)
            eval_score_pesq = pesq(16000, y, y_p, 'wb')

            stoi_sum+= eval_score_stoi
            pesq_sum += eval_score_pesq
            total_element+=1
        except Exception as e:
            print("Error at ", i, type(e).__name__)
            continue

        #print("Test2")
        if total_element % 100 == 1:
          stoi_score = (stoi_sum/total_element)*100
          pesq_score = pesq_sum/total_element
          print("Total element  : ",total_element)
          print("Stoi score : ", stoi_score)
          print("Pesq score : ", pesq_score)

    stoi_score = (stoi_sum/total_element)*100
    pesq_score = pesq_sum/total_element
    return stoi_score,pesq_score   


def main(args):
    self_sample_rate = 16_000
    total_seconds_of_audio = 10
    total_number_of_sample = self_sample_rate * total_seconds_of_audio

    model = Demucs(input_shape=(total_number_of_sample, 1))
    model.load_weights(args.model_path)
    print("Weight loaded")
    train_ds = get_dataset_from_tfrecords(args.tfrecord_path)
    print(train_ds)
    eval_score_stoi, eval_score_pesq = evaluate_stoi_pesq(train_ds,model)

    print("Evaluation score for stoi ", eval_score_stoi)
    print("Evaluation score for pesq ", eval_score_pesq)

   
if __name__== '__main__':
    main(parse_args())



#python3 evl.py --model_path /home/ml-dev/Noise_reducer/out/checkpoint_25_nov/checkpoint_epoch-018_loss-0.002110.h5 --tfrecord_path  /home/ml-dev/Data/Testing_noise_reducer/dns_2020_test_data/no_reverb/tfrecord/