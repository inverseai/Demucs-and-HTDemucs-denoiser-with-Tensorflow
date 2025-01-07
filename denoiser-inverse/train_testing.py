#!/usr/bin/env python3

"""import libraries"""
import tensorflow as tf
import argparse
import os
from denoiser.demucs import Demucs
from denoiser.stft_loss import custom_loss, loss_func
from denoiser.augment import augment
from denoiser.utils import save_model, write_history


def _parse_batch(record_batch, sample_rate, duration, split):
	n_samples = sample_rate * duration

	# Create a description of the features
	feature_description = {
		'noisy': tf.io.FixedLenFeature([n_samples], tf.float32),
		'clean': tf.io.FixedLenFeature([n_samples], tf.float32),
	}
	# Parse the input `tf.Example` proto using the dictionary above
	example = tf.io.parse_example(record_batch, feature_description)
	noisy, clean = tf.expand_dims(example['noisy'], axis=-1), tf.expand_dims(example['clean'], axis=-1)
	if split == 'train':
		noisy, clean = augment(noisy, clean)
	return noisy, clean


def get_dataset_from_tfrecords(tfrecords_dir='/content/sample_data/tfrecords', split='train',
							   batch_size=64, sample_rate=48000, duration=4, AUTOTUNE = tf.data.experimental.AUTOTUNE):
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
	ds = tf.data.TFRecordDataset(files_ds, compression_type='ZLIB', num_parallel_reads=AUTOTUNE)
	# Prepare batches
	ds = ds.batch(batch_size, drop_remainder=True)

	# Parse a batch into a dataset of [noisy, clean] pairs
	ds = ds.map(lambda x: _parse_batch(x, sample_rate, duration, split))
	ds = ds.repeat()
	# if split == 'train':
	return ds.prefetch(buffer_size=AUTOTUNE)

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--preload', type=str, dest='preload',
						default= 'None',
						help='If preload is not None, model + architecture will be load from the location '
							 '(default: %(default)s)')
	parser.add_argument('--autotune', type=int, dest='AUTOTUNE',
						default= -1,
						help='some default tweaking system '
						'(default: %(default)s)')

	parser.add_argument('--sample_rate', type=int, dest='sample_rate',
						default= 48_000,
						help='sample rate of audio data '
						'(default: %(default)s)')

	parser.add_argument('--total_seconds_of_audio', type=int, dest='total_seconds_of_audio',
						default= 4,
						help='total seconds of audio data in each data sample in noisy/clean audio '
						'(default: %(default)s)')

	parser.add_argument('--batch_size', type=int, dest='batch_size',
						default= 16,
						help='batch size '
						'(default: %(default)s)')

	parser.add_argument('--learning_rate', type=float, dest='learning_rate',
						default= 3e-4,
						help='learning rate '
						'(default: %(default)s)')

	parser.add_argument('--beta_1', type=float, dest='beta_1',
						default= 0.9,
						help='beta_1 '
						'(default: %(default)s)')

	parser.add_argument('--beta_2', type=float, dest='beta_2',
						default= 0.999,
						help='beta_2 '
						'(default: %(default)s)')

	parser.add_argument('--epsilon', type=float, dest='epsilon',
						default= 1e-07,
						help='epsilon '
						'(default: %(default)s)')

	parser.add_argument('--tfrecords_dir', type=str, dest='tfrecords_dir',
						default= 'tfrecords',
						help='tfrecords directory '
						'(default: %(default)s)')

	parser.add_argument('--steps', type=int, dest='steps',
						default= 1,
						help='we run model.fit these number of steps '
						'(default: %(default)s)')

	parser.add_argument('--steps_per_epoch', type=int, dest='steps_per_epoch',
						default= 20,
						help='we pass steps_per_epoch number of batches in each epoch '
						'(default: %(default)s)')

	parser.add_argument('--num_epoch_per_step', type=int, dest='num_epoch_per_step',
						default= 5,
						help='number of epoch we will run per step '
						'(default: %(default)s)')

	parser.add_argument('--saved_model_base_name', type=str, dest='saved_model_base_name',
						default= 'saved_weight_',
						help='saved model base name '
						'(default: %(default)s)')

	parser.add_argument('--saved_model_location', type=str, dest='saved_model_location',
						default= './outputs/saved_model',
						help='saved model directory '
						'(default: %(default)s)')

	parser.add_argument('--history_path', type=str, dest='history_path',
						default= './outputs/history.json',
						help='history directory '
						'(default: %(default)s)')
	parser.add_argument('--distributed', type=int, dest='distributed',
						default= 1,
						help='if 1, the code will do distributed training, else it will run on single device '
						'(default: %(default)s)')
	return parser.parse_args()



def get_compiled_model(args, total_number_of_sample, BATCH_SIZE):
	# Make a simple 2-layer densely-connected neural network.
	model = Demucs(input_shape=(total_number_of_sample, 1))
	model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate, beta_1=args.beta_1, beta_2=args.beta_2, epsilon=args.epsilon), loss = custom_loss(BATCH_SIZE))
	return model

def make_or_restore_model(args, total_number_of_sample, BATCH_SIZE, checkpoint_dir, model = None):
	# Either restore the latest model, or create a fresh one
	# if there is no checkpoint available.
	if not isinstance(model, type(None)):
		return model
	checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
	if checkpoints:
		latest_checkpoint = max(checkpoints, key=os.path.getctime)
		print("Restoring from", latest_checkpoint)
		return tf.keras.models.load_model(latest_checkpoint, custom_objects={'custom_loss': custom_loss, "BATCH_SIZE": BATCH_SIZE})
	print("Creating a new model")
	return get_compiled_model(args, total_number_of_sample, BATCH_SIZE)

def get_dataset(args, BATCH_SIZE):
	train_dataset =  get_dataset_from_tfrecords(tfrecords_dir=args.tfrecords_dir,batch_size=BATCH_SIZE, AUTOTUNE=args.AUTOTUNE)
	test_dataset= get_dataset_from_tfrecords(tfrecords_dir=args.tfrecords_dir,batch_size=BATCH_SIZE, split='test', AUTOTUNE=args.AUTOTUNE)
	valid_dataset= get_dataset_from_tfrecords(tfrecords_dir=args.tfrecords_dir,batch_size=BATCH_SIZE, split='validate', AUTOTUNE=args.AUTOTUNE)
	return train_dataset, test_dataset, valid_dataset

def run_training(args, total_number_of_sample, checkpoint_dir, ckpt = 'ckpt', model = None):
	print('\n\n\n\n\n\n\n\n\nstart new run training task')
	# Create a MirroredStrategy.
	strategy = tf.distribute.MirroredStrategy()
	BATCH_SIZE_PER_REPLICA = args.batch_size
	BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

	# Open a strategy scope and create/restore the model
	with strategy.scope():
		model = make_or_restore_model(args, total_number_of_sample, BATCH_SIZE, checkpoint_dir, model)

	callbacks = [
		# This callback saves a SavedModel every epoch
		# We include the current epoch in the folder name.
		tf.keras.callbacks.ModelCheckpoint(
			filepath=checkpoint_dir + "/"+ ckpt + "-{epoch}", save_freq="epoch"
		)
	]
	train_dataset, test_dataset, valid_dataset = get_dataset(args, BATCH_SIZE)
	history = model.fit(train_dataset, epochs=args.num_epoch_per_step, callbacks=callbacks, validation_data = valid_dataset, steps_per_epoch = args.steps_per_epoch, validation_steps = args.steps_per_epoch, verbose=2)
	print('done\n\n')
	return model


if __name__ == '__main__':
	args = parse_args()
	sample_rate = args.sample_rate
	total_seconds_of_audio = args.total_seconds_of_audio
	total_number_of_sample = sample_rate * total_seconds_of_audio

	checkpoint_dir = "/home/fahim/noise-reducer-ml/denoiser-inverse/outputs/ckpt"
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)
	model = None
	model = run_training(args,total_number_of_sample, checkpoint_dir, 'ckpt', model)
	model = run_training(args,total_number_of_sample, checkpoint_dir, 'ckpt2', model)
	model = run_training(args,total_number_of_sample, checkpoint_dir, 'ckpt3', model)
	model = run_training(args,total_number_of_sample, checkpoint_dir, 'ckpt4', model)
