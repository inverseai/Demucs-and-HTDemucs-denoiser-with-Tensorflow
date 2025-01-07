#!/usr/bin/env python3

"""import libraries"""
import tensorflow as tf
import argparse
import os


from denoiser.demucs import Demucs
from denoiser.stft_loss import custom_loss, loss_func, l1_mrstft_loss
from denoiser.augment import augment
from denoiser.utils import save_model, write_history
import re

def _parse_batch(record_batch, sample_rate, duration, split):
	n_samples = sample_rate * duration

	# Create a description of the features
	feature_description = {
		'noisy': tf.io.FixedLenFeature([n_samples], tf.float32),
		'clean': tf.io.FixedLenFeature([n_samples], tf.float32),
	}
	# Parse the input `tf.Example` proto using the dictionary above
	example = tf.io.parse_example(record_batch, feature_description)
	#print("Len ",example)
	noisy, clean = tf.expand_dims(example['noisy'], axis=-1), tf.expand_dims(example['clean'], axis=-1)
	
	if split == 'train':
		noisy, clean = augment(noisy, clean)
	return noisy, clean
	# return clean, noisy


def get_dataset_from_tfrecords(args, tfrecords_dir='/content/sample_data/tfrecords', split='train',
							   batch_size=64, sample_rate=16000, duration=4, AUTOTUNE = tf.data.experimental.AUTOTUNE):
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
	if args.steps_per_epoch != -1 : ds = ds.repeat()
	# if split == 'train':
	return ds.prefetch(buffer_size=AUTOTUNE)

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--autotune', type=int, dest='AUTOTUNE',
						default= -1,
						help='some default tweaking system '
						'(default: %(default)s)')

	parser.add_argument('--sample_rate', type=int, dest='sample_rate',
						default=16_000,
						help='sample rate of audio data '
						'(default: %(default)s)')

	parser.add_argument('--total_seconds_of_audio', type=int, dest='total_seconds_of_audio',
						default= 10,
						help='total seconds of audio data in each data sample in noisy/clean audio '
						'(default: %(default)s)')

	parser.add_argument('--batch_size', type=int, dest='batch_size',
						default= 1,
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

	parser.add_argument('--steps_per_epoch', type=int, dest='steps_per_epoch',
						default= -1,
						help='we pass steps_per_epoch number of batches in each epoch, -1 to run over all batch '
						'(default: %(default)s)')
	parser.add_argument('--val_steps_per_epoch', type=int, dest='val_steps_per_epoch',
						default= -1,
						help='we pass val_steps_per_epoch number of batches in each epoch, -1 to run over all batch '
						'(default: %(default)s)')
	parser.add_argument('--val_freq', type=int, dest='val_freq',
						default= 0,
						help='we pass val_freq, 0 means no validation will happen '
						'(default: %(default)s)')

	parser.add_argument('--epochs', type=int, dest='epochs',
						default= 5,
						help='number of epoch we will run per step '
						'(default: %(default)s)')

	parser.add_argument('--saved_model_base_name', type=str, dest='saved_model_base_name',
						default= 'saved_weight_',
						help='saved model base name '
						'(default: %(default)s)')

	parser.add_argument('--checkpoints_dir', type=str, dest='checkpoints_dir',
						default= './outputs/checkpoints/',
						help='checkpoints directory '
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



def get_compiled_model(args, BATCH_SIZE, compiled=True):
	# Make a simple 2-layer densely-connected neural network.
	# model = Demucs(input_shape=(args.sample_rate * args.total_seconds_of_audio, 1), kernel_initializer=tf.keras.initializers.GlorotUniform)
	model = Demucs(input_shape=(args.sample_rate * args.total_seconds_of_audio, 1))
	print(model.summary())
	if compiled:
		model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate, beta_1=args.beta_1, beta_2=args.beta_2, epsilon=args.epsilon), loss = custom_loss(BATCH_SIZE))
	return model

def make_or_restore_model(args, BATCH_SIZE, compiled=True):
	# Either restore the latest model, or create a fresh one
	# if there is no checkpoint available.
	checkpoints = [os.path.join(args.checkpoints_dir, name) for name in os.listdir(args.checkpoints_dir)]
	
	print("Created a new model")
	if checkpoints:
		latest_checkpoint = max(checkpoints, key=os.path.getctime)
		print("Restoring from", latest_checkpoint)
		#model = get_compiled_model(args, BATCH_SIZE, compiled=compiled)
		return tf.keras.models.load_model(latest_checkpoint, custom_objects={'custom_loss': custom_loss, "BATCH_SIZE": BATCH_SIZE})
		#model.load_weights(latest_checkpoint)
	else:
		model = get_compiled_model(args, BATCH_SIZE, compiled=compiled)
	return model

def get_dataset(args, BATCH_SIZE):
	train_dataset =  get_dataset_from_tfrecords(
			args,
			tfrecords_dir=args.tfrecords_dir,
			batch_size=BATCH_SIZE,
			sample_rate=args.sample_rate,
			duration=args.total_seconds_of_audio,
			AUTOTUNE=(tf.data.AUTOTUNE if args.AUTOTUNE == -1 else args.AUTOTUNE)
		)
	# test_dataset= get_dataset_from_tfrecords(
	# 		args,
	# 		tfrecords_dir=args.tfrecords_dir,
	# 		batch_size=BATCH_SIZE,
	# 		split='test',
	# 		sample_rate=args.sample_rate,
	# 		duration=args.total_seconds_of_audio,
	# 		AUTOTUNE=(tf.data.experimental.AUTOTUNE if args.AUTOTUNE == -1 else args.AUTOTUNE)
	# 	)
	# valid_dataset= get_dataset_from_tfrecords(
	# 		args,
	# 		tfrecords_dir=args.tfrecords_dir,
	# 		batch_size=BATCH_SIZE,
	# 		split='validate',
	# 		sample_rate=args.sample_rate,
	# 		duration=args.total_seconds_of_audio,
	# 		AUTOTUNE=(tf.data.experimental.AUTOTUNE if args.AUTOTUNE == -1 else args.AUTOTUNE)
	#  )
	return train_dataset



def get_dataset_val(args, BATCH_SIZE):
	valid_dataset= get_dataset_from_tfrecords(
			args,
			tfrecords_dir=args.tfrecords_dir,
			batch_size=BATCH_SIZE,
			split='validate',
			sample_rate=args.sample_rate,
			duration=args.total_seconds_of_audio,
			AUTOTUNE=(tf.data.experimental.AUTOTUNE if args.AUTOTUNE == -1 else args.AUTOTUNE)
	 )
	return valid_dataset

   #self.model.save(filepath, overwrite=True, include_optimizer=self.save_optimizer)

def run_training(args, steps = "ckpt"):
	# Create a MirroredStrategy.
	strategy = tf.distribute.MirroredStrategy()
	BATCH_SIZE_PER_REPLICA = args.batch_size
	BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
	print("Batch size", BATCH_SIZE)
	print(f'Number of GPUs we have: {strategy.num_replicas_in_sync}')
	# Open a strategy scope and create/restore the model
	validation_dataset = get_dataset_val(args, BATCH_SIZE)
	with strategy.scope():
		model = make_or_restore_model(args, BATCH_SIZE_PER_REPLICA)
		train_dataset = get_dataset(args, BATCH_SIZE)
		train_dataset = strategy.experimental_distribute_dataset(train_dataset)

		if args.val_freq==1:
			callbacks = [
				# This callback saves a SavedModel every epoch.We include the current epoch in the folder name.
				tf.keras.callbacks.ModelCheckpoint(
				filepath=os.path.join(args.checkpoints_dir , steps + "_epoch-{epoch:03d}_loss-{loss:.6f}_val_loss-{val_loss:.6f}.h5"),
				save_best_only=False,
				save_weights_only=False
				)
			]
		else:
			callbacks = [
				tf.keras.callbacks.ModelCheckpoint(
				filepath=os.path.join(args.checkpoints_dir , steps + "_epoch-{epoch:03d}_loss-{loss:.6f}.h5"),
				save_best_only=False,
				save_weights_only=False
				)
			]
		
		if args.val_freq ==0: 
			history = model.fit(train_dataset, epochs=args.epochs, callbacks=callbacks,  steps_per_epoch = (None if args.steps_per_epoch == -1 else args.steps_per_epoch), verbose = 1)
		else: 
			history = model.fit(train_dataset, epochs=args.epochs,  callbacks=callbacks, validation_data = validation_dataset, steps_per_epoch = (None if args.steps_per_epoch == -1 else args.steps_per_epoch),validation_steps = args.val_steps_per_epoch, validation_freq = args.val_freq, verbose = 1)
	print('done\n\n')
	return model

if __name__ == '__main__':
	args = parse_args()

	if not os.path.exists(args.checkpoints_dir):
		os.makedirs(args.checkpoints_dir)
	
	model = run_training(args, f'checkpoint')


# python3 /home/ml-dev/Noise_reducer/Noise_reducer_ml_latest/noise-reducer-ml/Train/denoiser-inverse/train.py  --autotune 32 --tfrecords_dir /media/ml-dev/1TB_volume/tfrecords_22_nov_2023/ --checkpoints_dir /home/ml-dev/Noise_reducer/out/checkpoint_25_nov/ --epochs 1000 --steps_per_epoch 11250 --batch_size 16 --learning_rate 1e-4 --sample_rate 16000 --total_seconds_of_audio 10 >> /home/ml-dev/Noise_reducer/out/every_iteration_log/train_log_1e-4_after_p.txt
# python3 /home/ml-dev/Noise_reducer/Noise_reducer_ml_latest/noise-reducer-ml/Train/denoiser-inverse/train.py  --autotune 32 --tfrecords_dir /media/ml-dev/1TB_volume/tfrecords_22_nov_2023/ --checkpoints_dir /home/ml-dev/Noise_reducer/out/checkpoint_25_nov/ --epochs 1000 --steps_per_epoch 11250 --val_steps_per_epoch 1125 --val_freq 1 --batch_size 16 --learning_rate 1e-4 --sample_rate 16000 --total_seconds_of_audio 10  >> /home/ml-dev/Noise_reducer/out/every_iteration_log/train_log_1e-4_after_pp.txt