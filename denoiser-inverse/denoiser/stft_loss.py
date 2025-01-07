import tensorflow as tf

def stft(x, fft_size, hop_size, win_length, window_fn):
	"""Perform STFT and convert to magnitude spectrogram.
	Args:
		x: signal (B, sz), B = number of batchs, sz = length of audio, channel is squeezed
		fft_size (int): FFT size. fft_length
		hop_size (int): Hop size. frame_step
		win_length (int): Window length. frame_length
		window_fn (func): Window function type.
	Returns:
		Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1). (B, #frames, #freq_bins)
		https://github.com/facebookresearch/denoiser/blob/c200d3eda590560f7180c625a86141182b4306dd/denoiser/stft_loss.py#L17
	"""
	x_stft = tf.signal.stft(signals = x, fft_length = fft_size, frame_step = hop_size, frame_length = win_length, window_fn = window_fn)
	real = tf.math.real(x_stft)
	imag = tf.math.imag(x_stft)
	# NOTE(kan-bayashi): clamp is needed to avoid nan or inf
	return tf.math.sqrt(tf.math.maximum(real**2 + imag**2, 1e-7))

def SpectralConvergengeLoss(y_true_mag, y_pred_mag):
	"""Spectral convergence loss module.
	Args:
		y_true_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
		y_pred_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
	Returns:
		Tensor: Spectral convergence loss value. (B, )
	"""
	return tf.norm(y_true_mag - y_pred_mag, ord="fro", axis=[-2, -1]) / tf.norm(y_true_mag, ord="fro", axis=[-2, -1])
	
def LogSTFTMagnitudeLoss(y_true_mag, y_pred_mag):
	"""Log STFT magnitude loss module.
	Args:
		y_true_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
		y_pred_mag (Tensor): Magnitude spectrogram of predicted   signal (B, #frames, #freq_bins).
	Returns:
		Tensor: Log STFT magnitude loss value. (B,)
	"""
	return tf.keras.metrics.mean_absolute_error(
		tf.math.log(tf.reshape(y_true_mag, [-1, y_true_mag.shape[1] * y_true_mag.shape[2]])), 
		tf.math.log(tf.reshape(y_pred_mag, [-1, y_pred_mag.shape[1] * y_pred_mag.shape[2]]))
		)

def STFTLoss(y_true, y_pred, fft_size=1024, shift_size=120, win_length=600, window="hann_window"):
	"""STFT loss method
	Args:
		y_true, y_pred: Groundtruth and predicted signal (B, sz), B = number of batchs, sz = length of audio, channel is squeezed
		fft_size (int):
		hop_sizes (int):
		win_length (int):
		window (string):
	returns:
		Tensor: spectral convergence loss value. (B,)
		Tensor: log STFT magnitude loss value. (B,)
	"""
	window_fn = getattr(tf.signal, window)
	y_true_mag = stft(y_true, fft_size, shift_size, win_length, window_fn) # (B, #frames, fft_size // 2 + 1).
	y_pred_mag = stft(y_pred, fft_size, shift_size, win_length, window_fn) # (B, #frames, fft_size // 2 + 1).
	sc_loss = SpectralConvergengeLoss(y_true_mag, y_pred_mag) # (B,)
	mag_loss = LogSTFTMagnitudeLoss(y_true_mag, y_pred_mag) # (B,)

	return sc_loss, mag_loss

def MultiResolutionSTFTLoss(y_true, y_pred, fft_sizes=[1024, 2048, 512], hop_sizes=[120, 240, 50], win_lengths=[600, 1200, 240], window="hann_window", factor_sc=0.1, factor_mag=0.1):
	"""Multi resolution stft loss method
	Args:
		y_true, y_pred: Groundtruth and predicted signal (B, sz), B = number of batchs, sz = length of audio, channel is squeezed
		fft_sizes (list): List of fft sizes
		hop_sizes (list): List of hop sizes
		win_lengths (list): List of window lengths.
		window (string): window function type
		factor (float), factor_sc and factor_mag: a balancing factor across different losses
	returns:
		Tensor: Multi resolution spectral convergence loss value. (B,)
		Tensor: Multi resolution log STFT magnitude loss value. (B,)
	"""

	sc_loss, mag_loss, fft_sizes_len = 0.0, 0.0, len(fft_sizes)
	"""assert fft_sizes_len == len(hop_sizes) and fft_sizes_len == len(win_lengths)"""
	for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
		sc_l, mag_l = STFTLoss(y_true, y_pred, fs, ss, wl, window) # (B,), (B,)
		sc_loss += sc_l
		mag_loss += mag_l
	sc_loss /= fft_sizes_len
	mag_loss /= fft_sizes_len
	return factor_sc * sc_loss, factor_mag * mag_loss

def l1_mrstft_loss(y_true, y_pred):
	"""calculate l1(mae) + multi resolution stft loss
	Args:
		y_true: (B, sz) sz = length of audio, channel is squeezed
		y_pred (B, sz) sz = length of audio, channel is squeezed
	Returns:
		Tensor: Multi resolution stft loss + l1(mae) loss of (B,) shape
	"""
	if len(y_true.shape) == 3:
		y_true = tf.squeeze(y_true, [-1])
		y_pred = tf.squeeze(y_pred, [-1])
	loss = tf.keras.metrics.mean_absolute_error(y_true, y_pred) # (B,) shape
	# sc_loss, mag_loss = MultiResolutionSTFTLoss(y_true, y_pred) # (B,) shape
	# loss = loss + sc_loss + mag_loss # (B,) shape
	return loss

class custom_loss(tf.keras.losses.Loss):
	def __init__(self, BATCH_SIZE = -1, extra = 0.0, **kwargs):
		super(custom_loss, self).__init__(**kwargs)
		self.BATCH_SIZE = BATCH_SIZE
		#self.Model_size = extra

	def call(self, y_true, y_pred):
		loss = l1_mrstft_loss(tf.squeeze(y_true, [-1]), tf.squeeze(y_pred, [-1])) #+ self.Model_size
		return tf.nn.compute_average_loss(loss, global_batch_size=self.BATCH_SIZE) #+ self.Model_size
		# loss = tf.reduce_sum(loss) * (1. / self.BATCH_SIZE)
		# return loss

	def get_config(self):
		config = super().get_config().copy()
		config.update({'BATCH_SIZE': self.BATCH_SIZE})
		return config
	

def loss_func(y_true, y_pred):
	loss = l1_mrstft_loss(tf.squeeze(y_true, [-1]), tf.squeeze(y_pred, [-1]))
	loss = tf.reduce_mean(loss)
	return loss