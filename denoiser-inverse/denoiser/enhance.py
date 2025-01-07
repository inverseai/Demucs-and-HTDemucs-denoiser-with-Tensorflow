import tensorflow as tf
from demucs import Demucs

def pred(model = None,model_path = '', noisy_file_path = '/content/sample_data/p287_006.wav'):
	self_sample_rate = 16_000
	total_seconds_of_audio = 4
	total_number_of_sample = self_sample_rate * total_seconds_of_audio
	if isinstance(model, type(None)):
		model = Demucs(input_shape=(total_number_of_sample, 1))
		model.load_weights(model_path)
		#model = tf.keras.models.load_model(model_path, custom_objects={'custom_loss': custom_loss})
	raw_audio_noisy = tf.io.read_file(noisy_file_path)
	noisy, sample_rate = tf.audio.decode_wav( raw_audio_noisy, desired_channels=1)
	number_of_total_sample = noisy.shape[0]
	number_of_chunks = (number_of_total_sample + total_number_of_sample - 1) // total_number_of_sample
	number_of_total_sample_upperbound = number_of_chunks * total_number_of_sample
	padded_noisy = tf.pad(noisy, tf.constant([[0, number_of_total_sample_upperbound - number_of_total_sample], [0, 0]]), "CONSTANT")
	padded_noisy = tf.reshape(padded_noisy, [number_of_total_sample_upperbound // total_number_of_sample, total_number_of_sample, 1])
	estimate = model.predict(padded_noisy)
	estimate = tf.reshape(estimate, [number_of_total_sample_upperbound, 1])
	estimate = estimate[:number_of_total_sample,:]
	return tf.reshape(noisy,[-1]), tf.reshape(estimate,[-1])


