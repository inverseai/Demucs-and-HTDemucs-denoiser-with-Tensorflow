import argparse
import math
import os

import numpy as np
import pandas as pd
import tensorflow as tf
import json
from tqdm import tqdm
import multiprocessing
import time




_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_DEFAULT_NOISY_JSON = os.path.join(_BASE_DIR, 'noisy.json')
_DEFAULT_CLEAN_JSON = os.path.join(_BASE_DIR, 'clean.json')
_DEFAULT_OUTPUT_DIR = os.path.join(_BASE_DIR, 'tfrecords')

_DEFAULT_DURATION = 10  # seconds
_DEFAULT_SAMPLE_RATE = 16000

_DEFAULT_TEST_SIZE = 0.1
_DEFAULT_VAL_SIZE = 0.1

_DEFAULT_NUM_SHARDS_TRAIN = 16
_DEFAULT_NUM_SHARDS_TEST = 2
_DEFAULT_NUM_SHARDS_VAL = 2

_SEED = 2020
_CORE = 1

def load_file_location_from_json_file(data_path):
	with open(data_path, "r") as fp:
		data = json.load(fp)
	return data

def _float_feature(list_of_floats):  # float32
	return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

class TFRecordsConverter:
	"""Convert audio to TFRecords."""
	def __init__(self, noisy_json_location, clean_json_location, output_dir, n_shards_train, n_shards_test, n_shards_val, duration, sample_rate, test_size, val_size, core):
		self.output_dir = output_dir
		self.n_shards_train = n_shards_train
		self.n_shards_test = n_shards_test
		self.n_shards_val = n_shards_val
		self.duration = duration
		self.sample_rate = sample_rate

		if not os.path.exists(self.output_dir):
			os.makedirs(self.output_dir)

		self.noisy_location = load_file_location_from_json_file(noisy_json_location)
		self.clean_location = load_file_location_from_json_file(clean_json_location)
		# Shuffle data by "sampling" the entire data-frame
		# need to shuffle noisy_location and clean_location
		n_samples = len(self.noisy_location)
		self.n_test = math.ceil(n_samples * test_size)
		self.n_val = math.ceil(n_samples * val_size)
		self.n_train = n_samples - self.n_test - self.n_val
		self.operation = []
		self.core = core

	def _get_shard_path(self, split, shard_id, shard_size):
		return os.path.join(self.output_dir,
							'{}-{:03d}-{}.tfrecord'.format(split, shard_id,
														   shard_size))

	def _write_tfrecord_file(self, shard_path, indices):
		"""Write TFRecord file."""
		with tf.io.TFRecordWriter(shard_path, options='ZLIB') as out:
			for index in indices:
				noisy_file_path = self.noisy_location[index]
				clean_file_path = self.clean_location[index]
				try:
					raw_audio_noisy = tf.io.read_file(noisy_file_path)
					noisy, sample_rate = tf.audio.decode_wav( raw_audio_noisy, desired_channels=1, desired_samples=self.sample_rate * self.duration)
				except Exception as e:
					print(noisy_file_path)
					print(e)
				try:
					raw_audio_clean = tf.io.read_file(clean_file_path)
					clean, _sample_rate = tf.audio.decode_wav( raw_audio_clean, desired_channels=1, desired_samples=self.sample_rate * self.duration)
				except Exception as e:
					print(clean_file_path)
					print(e)
				example = tf.train.Example(features=tf.train.Features(feature={
					'noisy': _float_feature(noisy.numpy().flatten().tolist()),
					'clean': _float_feature(clean.numpy().flatten().tolist())
					}))
				out.write(example.SerializeToString())

	def write_shard(self, l, r):
		for i in range(l, r+1):
			shard_path, s, e = self.operation[i]
			file_indices = np.arange(s, e)
			self._write_tfrecord_file(shard_path, file_indices)

	def convert(self):
		"""Convert to TFRecords.

		Partition data into training, testing and validation sets. Then,
		divide each data set into the specified number of TFRecords shards.
		"""
		splits = ('train', 'test', 'validate')
		split_sizes = (self.n_train, self.n_test, self.n_val)
		split_n_shards = (self.n_shards_train, self.n_shards_test,
						  self.n_shards_val)

		offset = 0
		for split, size, n_shards in zip(splits, split_sizes, split_n_shards):
			if n_shards == 0: continue
			print('Converting {} set into TFRecord shards...'.format(split))
			shard_size = math.ceil(size / n_shards)
			cumulative_size = offset + size
			for shard_id in tqdm(range(1, n_shards + 1)):
				step_size = min(shard_size, cumulative_size - offset)
				shard_path = self._get_shard_path(split, shard_id, step_size)
				# Generate a subset of indices to select only a subset of
				# audio-files/labels for the current shard.
				# file_indices = np.arange(offset, offset + step_size)
				self.operation.append((shard_path, offset, offset+step_size))
				# self._write_tfrecord_file(shard_path, file_indices)
				offset += step_size
		processes = []
		sz = len(self.operation)
		step = (sz + self.core - 1) // self.core
		start = 0
		for _ in range(self.core):
			curr_step = min(step, sz-start)
			p = multiprocessing.Process(target=self.write_shard, args=[start, start + curr_step - 1])
			p.start()
			processes.append(p)
			start += curr_step
		for _ in processes:
			_.join()
		print('Number of training examples: {}'.format(self.n_train))
		print('Number of testing examples: {}'.format(self.n_test))
		print('Number of validation examples: {}'.format(self.n_val))
		print('TFRecord files saved to {}'.format(self.output_dir))


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-n', '--noisy-json-location', type=str, dest='noisy_json',
						default=_DEFAULT_NOISY_JSON,
						help='File containing noisy audio file-paths '
							 '(default: %(default)s)')
	parser.add_argument('-c', '--clean-json-location', type=str, dest='clean_json',
						default=_DEFAULT_CLEAN_JSON,
						help='File containing clean audio file-paths '
							 '(default: %(default)s)')
	parser.add_argument('-o', '--output-dir', type=str, dest='output_dir',
						default=_DEFAULT_OUTPUT_DIR,
						help='Output directory to store TFRecord files.'
							 '(default: %(default)s)')
	parser.add_argument('--num-shards-train', type=int,
						dest='n_shards_train',
						default=_DEFAULT_NUM_SHARDS_TRAIN,
						help='Number of shards to divide training set '
							 'TFRecords into. (default: %(default)s)')
	parser.add_argument('--num-shards-test', type=int,
						dest='n_shards_test',
						default=_DEFAULT_NUM_SHARDS_TEST,
						help='Number of shards to divide testing set '
							 'TFRecords into. (default: %(default)s)')
	parser.add_argument('--num-shards-val', type=int,
						dest='n_shards_val',
						default=_DEFAULT_NUM_SHARDS_VAL,
						help='Number of shards to divide validation set '
							 'TFRecords into. (default: %(default)s)')
	parser.add_argument('--duration', type=int,
						dest='duration',
						default=_DEFAULT_DURATION,
						help='The duration for the resulting fixed-length '
							 'audio-data in seconds. Longer files are '
							 'truncated. Shorter files are zero-padded. '
							 '(default: %(default)s)')
	parser.add_argument('--sample-rate', type=int,
						dest='sample_rate',
						default=_DEFAULT_SAMPLE_RATE,
						help='The _actual_ sample-rate of wav-files to '
							 'convert. Re-sampling is not yet supported. '
							 '(default: %(default)s)')
	parser.add_argument('--test-size', type=float,
						dest='test_size',
						default=_DEFAULT_TEST_SIZE,
						help='Fraction of examples in the testing set. '
							 '(default: %(default)s)')
	parser.add_argument('--val-size', type=float,
						dest='val_size',
						default=_DEFAULT_VAL_SIZE,
						help='Fraction of examples in the validation set. '
							 '(default: %(default)s)')
	parser.add_argument('--core', type=int,
						dest='core',
						default=_CORE,
						help='number of cores for multiprocessing '
							 '(default: %(default)s)')

	return parser.parse_args()


def main(args):
	print('\n\n\n',args,'\n\n\n')
	converter = TFRecordsConverter(args.noisy_json,
								   args.clean_json,
								   args.output_dir,
								   args.n_shards_train,
								   args.n_shards_test,
								   args.n_shards_val,
								   args.duration,
								   args.sample_rate,
								   args.test_size,
								   args.val_size,
								   args.core)
	start = time.perf_counter()
	converter.convert()
	finish = time.perf_counter()
	print(f'Finished in {round(finish-start, 2)} second(s)')


if __name__ == '__main__':
	main(parse_args())

 
#python3 convert_tf_record.py -n /home/ml-dev/Data/Testing_noise_reducer/dns_2020_test_data/no_reverb/noisy.json -c /home/ml-dev/Data/Testing_noise_reducer/dns_2020_test_data/no_reverb/clean.json -o  /home/ml-dev/Data/Testing_noise_reducer/dns_2020_test_data/no_reverb/ --num-shards-train 1 --num-shards-test 0 --num-shards-val 0 --sample-rate 16000 --test-size 0 --val-size 0