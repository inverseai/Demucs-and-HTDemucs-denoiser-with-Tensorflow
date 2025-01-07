import random
import tensorflow as tf

def _reverb(source, initial, first_delay, rt60, self_repeat, self_jitter, self_sample_rate):
	"""
	Return the reverb for a single source.
	"""
	length = source.shape[-2]
	reverb = tf.zeros_like(source)
	for _ in range(self_repeat):
		frac = 1  # what fraction of the first echo amplitude is still here
		echo = initial * source
		while frac > 1e-3:
			# First jitter noise for the delay
			jitter = 1 + self_jitter * random.uniform(-1, 1)
			delay = min( 1 + int(jitter * first_delay * self_sample_rate), length)
			# Delay the echo in time by padding with zero on the left
			echo = tf.pad(echo[:,:-delay,:], [[0, 0], [delay, 0],[0, 0]], "CONSTANT")
			reverb += echo

			# Second jitter noise for the attenuation
			jitter = 1 + self_jitter * random.uniform(-1, 1)
			# we want, with `d` the attenuation, d**(rt60 / first_ms) = 1e-3
			# i.e. log10(d) = -3 * first_ms / rt60, so that
			attenuation = 10**(-3 * jitter * first_delay / rt60)
			echo *= attenuation
			frac *= attenuation
	return reverb
def augment(noisy, clean):
	""" RevEcho
	Hacky Reverb but runs on GPU without slowing down training.
	This reverb adds a succession of attenuated echos of the input
	signal to itself_ Intuitively, the delay of the first echo will happen
	after roughly 2x the radius of the room and is controlled by `first_delay`.
	Then RevEcho keeps adding echos with the same delay and further attenuation
	until the amplitude ratio between the last and first echo is 1e-3.
	The attenuation factor and the number of echos to adds is controlled
	by RT60 (measured in seconds). RT60 is the average time to get to -60dB
	(remember volume is measured over the squared amplitude so this matches
	the 1e-3 ratio).
	At each call to RevEcho, `first_delay`, `initial` and `RT60` are
	sampled from their range. Then, to prevent this reverb from being too regular,
	the delay time is resampled uniformly within `first_delay +- 10%`,
	as controlled by the `jitter` parameter. Finally, for a denser reverb,
	multiple trains of echos are added with different jitter noises.
	Args:
		- initial: amplitude of the first echo as a fraction
			of the input signal. For each sample, actually sampled from
			`[0, initial]`. Larger values means louder reverb. Physically,
			this would depend on the absorption of the room walls.
		- rt60: range of values to sample the RT60 in seconds, i.e.
			after RT60 seconds, the echo amplitude is 1e-3 of the first echo.
			The default values follow the recommendations of
			https://arxiv.org/ftp/arxiv/papers/2001/2001.08662.pdf, Section 2.4.
			Physically this would also be related to the absorption of the
			room walls and there is likely a relation between `RT60` and
			`initial`, which we ignore here.
		- first_delay: range of values to sample the first echo delay in seconds.
			The default values are equivalent to sampling a room of 3 to 10 meters.
		- repeat: how many train of echos with differents jitters to add.
			Higher values means a denser reverb.
		- jitter: jitter used to make each repetition of the reverb echo train
			slightly different. For instance a jitter of 0.1 means
			the delay between two echos will be in the range `first_delay +- 10%`,
			with the jittering noise being resampled after each single echo.
		- keep_clean: fraction of the reverb of the clean speech to add back
			to the ground truth. 0 = dereverberation, 1 = no dereverberation.
		- sample_rate: sample rate of the input signals.
	"""
	self_proba = 0.5
	self_initial = 0.3
	self_rt60 = (0.3, 1.3)
	self_first_delay = (0.01, 0.03)
	self_repeat = 3
	self_jitter = 0.1
	self_keep_clean = 0.1
	self_sample_rate = 16000

	if random.random() >= self_proba:
		return noisy, clean
	noise = noisy - clean
	initial = random.random() * self_initial
	first_delay = random.uniform(*self_first_delay)
	rt60 = random.uniform(*self_rt60)
	reverb_noise = _reverb(noise, initial, first_delay, rt60, self_repeat, self_jitter, self_sample_rate)
	noise += reverb_noise
	reverb_clean = _reverb(clean, initial, first_delay, rt60, self_repeat, self_jitter, self_sample_rate)
	clean += self_keep_clean * reverb_clean
	noise += (1 - self_keep_clean) * reverb_clean
	noisy = noise + clean
	return noisy, clean