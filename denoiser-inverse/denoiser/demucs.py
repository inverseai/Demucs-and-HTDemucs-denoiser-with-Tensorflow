
import sys

sys.path.append('..')
# from Diffq import DiffQuantiazer
# from Custom_Diffq_Layers import Con1DLayer, Con2DLayer, DenseLayer, Con1DTranspose, LSTMCell
import tensorflow as tf
import math
import numpy as np
from tensorflow.python.ops.init_ops import Initializer

"""custom layers"""
"""custom weight initialization scaling for convolution layer"""


class ConvScaling(Initializer):
    def __init__(self, scale=1.0, reference=0.1, seed=None, dtype=tf.float32):
        self.scale = scale
        self.reference = reference
        self.seed = seed
        self.dtype = dtype

    def __call__(self, shape, dtype=None, partition_info=None):
        stdv = 1.0 / (shape[0] * shape[1])
        w = tf.random.uniform(shape, minval=-stdv, maxval=stdv, dtype=self.dtype, seed=self.seed)
        std = tf.math.reduce_std(w)
        scale = (std / self.reference) ** .5
        w = w / scale
        return w

    def get_config(self):
        return {
            'scale': self.scale,
            'seed': self.seed,
            'dtype': self.dtype.name
        }


"""gated linear unit activation layer"""


def glu(x):
    n_units = tf.shape(x)[-1] // 2
    return x[..., :n_units] * tf.nn.sigmoid(x[..., n_units:])


def sinc(t):
    return tf.where(t == 0, tf.constant(1., dtype=t.dtype), tf.math.sin(t) / t)


"""Bi Lstm Layer"""


class BLSTM:
    def __init__(self, dim, num_layers=2, bidirectional=True):
        self.dim = dim
        self.num_layers = num_layers
        self.lstm = tf.keras.Sequential()
        # self.lstm = tf.keras.layers.LSTM(dim, input_shape=(31,768))
        # self.lstm2 = tf.keras.layers.LSTM(dim)
        # self.dense = tf.keras.layers.Dense(dim)
        for _ in range(num_layers):
            if bidirectional:
                bidirectional_layer = tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(units=dim, activation='tanh', return_sequences=True))
                self.lstm.add(bidirectional_layer)
                self.linear = tf.keras.layers.Dense(dim)

            else:
                self.lstm.add(
                    tf.keras.layers.RNN(tf.keras.layers.LSTMCell(units=dim, activation='tanh'), return_sequences=True)
                )

    def __call__(self, x):
        x = self.lstm(x)
        if self.linear:
            x = self.linear(x)
        return x


"""upsampling and downsampling: change sample rate of the given audio with factor of 2 or 1/2"""


def kernel_for_upsample_and_downsample_2(zeros=56):
    win = tf.signal.hann_window(window_length=4 * zeros + 1, periodic=False)
    winodd = win[1::2]
    t = tf.linspace(-zeros + 0.5, zeros - 0.5, 2 * zeros)
    t *= math.pi
    # print(t)
    kernel = tf.reshape(sinc(t) * winodd, [-1, 1, 1])
    return kernel


def upsample2(x, zeros=56):
    batch, sample, ch = x.shape
    kernel = kernel_for_upsample_and_downsample_2(zeros)
    out = tf.reshape(x, [-1, sample, 1])
    # print(out.shape)
    out = tf.nn.conv1d(input=out, filters=kernel, stride=1, padding="SAME")
    # out = out[:,1:,:]
    out = tf.reshape(out, [-1, sample, ch])
    y = tf.stack([x, out], axis=-2)
    # print(tf.reshape(y, [-1, sample * 2, 1]))
    return tf.reshape(y, [-1, sample * 2, 1])


def downsample2(x, zeros=56):
    if x.shape[-2] % 2 != 0:
        x = tf.pad(x, [[0, 0], [0, 1], [0, 0]], "CONSTANT")
    xeven = x[:, ::2, :]
    xodd = x[:, 1::2, :]
    # print(xeven)
    batch, sample, ch = xodd.shape
    kernel = kernel_for_upsample_and_downsample_2(zeros)
    # print(kernel.shape)
    # kernel = tf.reshape(kernel, shape=(112, 1, 1))
    xodd = tf.reshape(xodd, [-1, sample, 1])
    xodd = tf.pad(xodd, [[0, 0, ], [1, 1], [0, 0]], "CONSTANT")
    # print(xodd)
    xodd = tf.nn.conv1d(input=xodd, filters=kernel, stride=1, padding="SAME")[:, :-2, :]
    # print(xodd)
    # xodd = xodd[:,:-1,:]
    out = xeven + tf.reshape(xodd, [-1, sample, ch])
    out = tf.reshape(out, [-1, sample, ch])
    out = out * 0.5
    return out


def valid_length(length, depth=5, s=4, k=8):
    length = math.ceil(length * 4)
    for _ in range(depth):
        length = math.ceil((length - k) / s) + 1
        length = max(length, 1)
    for _ in range(depth):
        length = (length - 1) * s + k
    length = int(math.ceil(length / 4))
    return int(length)


import keras.backend as K


def Demucs(input_shape, hidden=64, depth=5, sr=16_000):
    # assert depth == 5
    encoder = []
    decoder = []
    activation = tf.keras.layers.Lambda(glu)

    chin, chout = 1, 1
    for index in range(depth):
        encoder.append(
            tf.keras.Sequential(
                [
                    tf.keras.layers.Conv1D(
                        filters=hidden, kernel_size=8, strides=4, activation=tf.nn.relu
                    ),
                    tf.keras.layers.Conv1D(
                        filters=hidden * 2, kernel_size=1, strides=1
                    ),
                    activation
                ]
                , name=f"encode_{index}"
            ))
        decoder.append(tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(
                    filters=hidden * 2, kernel_size=1, strides=1
                ),
                activation,
                tf.keras.layers.Conv1DTranspose(
                    filters=chout, kernel_size=8, strides=4
                )
            ]
            + ([tf.keras.layers.ReLU()] if index else [])
            , name=f"decode_{index}"
        ))
        chout, chin, hidden = hidden, hidden, hidden * 2

    decoder = decoder[::-1]
    bidirectional = True
    lstm = BLSTM(chin, bidirectional)
    input_audio = tf.keras.layers.Input(shape=input_shape, name="input_layer")
    x = input_audio

    """normalize"""
    # mono = tf.math.reduce_mean(x,axis=1,keepdims=True)
    std = tf.math.reduce_std(x, axis=1, keepdims=True)
    x = x / (1e-3 + std)
    length = x.shape[1]
    """pad to have valid length"""
    x = tf.pad(x, [[0, 0], [0, valid_length(length) - length], [0, 0]], "CONSTANT")

    """upsample the samplerate by 4"""
    x = upsample2(x)
    # tf.print(x.shape)
    x = upsample2(x)
    # x = upsample2(x)

    """encoder part"""
    skips = []  # used for skip connection
    for encode in encoder:
        x = encode(x)
        # print(x.shape)
        val = tf.identity(x)
        skips.append(val)
    # print(skips)
    x = lstm(x)
    # x = tf.keras.layers.LSTM(chin)(x)

    """decoder part"""
    for decode in decoder:
        skip = skips.pop(-1)
        # print(skip.shape)
        # print("Before", skip)
        # print("After", skip[:,:x.shape[1],:])
        x = tf.add(x, skip)
        # print(x)
        # x = x + skip[:,:x.shape[1],:] # add skip connection
        x = decode(x)

    # print(x.shape)
    """downsample the samplerate by 4"""
    x = downsample2(x)
    x = downsample2(x)
    # print(x.shape)
    # x = downsample2(x)

    x = x[:, :length, :]
    x = x * std  # multiply with normalize factor

    model = tf.keras.models.Model(input_audio, x, name='demucs')
    return model
