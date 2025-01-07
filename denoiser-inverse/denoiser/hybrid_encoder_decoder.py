import numpy as np
import tensorflow as tf
from keras.layers import (
    Layer,
    Conv1D,
    Conv2D,
    Conv1DTranspose,
    GroupNormalization,
    Activation,
    Conv2DTranspose,
)
from keras import models, Model, activations


class GLU(Layer):
    """Gated Linear Unit (GLU) activation function.

    Args:
        input_tensor: Input tensor.
        axis: The dimension on which to split the input tensor. Default: -1.

    Returns:
        Output tensor after applying GLU.
    """

    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, input_tensor):
        if input_tensor.shape[self.axis] % 2 != 0:
            raise ValueError(
                "The size of the input tensor along the specified axis must be even."
            )
        1
        a, b = tf.split(input_tensor, num_or_size_splits=2, axis=self.axis)
        return a * tf.sigmoid(b)


class LayerScale_old(Layer):
    """Layer scale from [Touvron et al 2021] (https://arxiv.org/pdf/2103.17239.pdf).
    This rescales diagonaly residual outputs close to 0 initially, then learnt."""

    def __init__(self, channels: int, init: float = 0, channel_last=True, **kwargs):
        """
        channel_last = False corresponds to (B, C, T) tensors
        channel_last = True corresponds to (T, B, C) tensors
        """
        super().__init__(**kwargs)
        self.channel_last = channel_last
        self.scale = tf.Variable(
            initial_value=tf.fill((init,), channels), trainable=True
        )

    def call(self, x):
        if self.channel_last:
            return self.scale * x
        else:
            return self.scale[:, tf.newaxis] * x


class LayerScale(Layer):
    """
    Layer scale from [Touvron et al 2021] (https://arxiv.org/pdf/2103.17239.pdf).
    This rescales diagonaly residual outputs close to 0 initially, then learnt.
    """

    def __init__(self, channels: int, init: float = 0, channel_last=True):
        """
        channel_last = False corresponds to (B, C, T) tensors
        channel_last = True corresponds to (T, B, C) tensors
        """
        super(LayerScale, self).__init__()
        self.channel_last = channel_last
        # Initialize the scale parameter as a learnable variable
        self.scale = self.add_weight(
            name="scale",
            shape=[channels],
            initializer=tf.constant_initializer(init),
            trainable=True,
        )

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        if self.channel_last:
            return self.scale * inputs
        else:
            # Reshape scale for broadcasting
            return tf.reshape(self.scale, [-1, 1]) * inputs


class IdentityNorm(Layer):
    """Identity normalization layer."""
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return inputs


class DConv(Model):
    """
    New residual branches in each encoder layer.
    This alternates dilated convolutions, potentially with LSTMs and attention.
    Also before entering each residual branch, dimension is projected on a smaller subspace,
    e.g. of dim `channels // compress`.
    """

    def __init__(
        self,
        channels,
        compress=4,
        depth=2,
        init=1e-4,
        norm=True,
        attn=False,
        heads=4,
        ndecay=4,
        lstm=False,
        gelu=True,
        kernel=3,
        dilate=True,
    ):
        super(DConv, self).__init__()

        assert kernel % 2 == 1
        self.channels = channels
        self.compress = compress
        self.depth = abs(depth)
        dilate = depth > 0

        norm_fn = IdentityNorm
        if norm:
            norm_fn = GroupNormalization

        hidden = channels // compress

        def get_activation():
            if gelu:
                return activations.gelu
            else:
                return activations.relu

        selected_activation = get_activation()

        self.layers_list = []
        for d in range(self.depth):
            dilation_rate = 2**d if dilate else 1
            padding = "same"  # 'same' padding for Keras to handle padding automatically
            mods = [
                Conv1D(
                    hidden,
                    kernel_size=kernel,
                    dilation_rate=dilation_rate,
                    padding=padding,
                ),
                norm_fn(1),
                Activation(selected_activation),
                Conv1D(2 * channels, kernel_size=1),
                norm_fn(1),
                GLU(axis=-1),
                LayerScale(channels, init, channel_last=True),
            ]

            self.layers_list.append(models.Sequential(mods))

    def call(self, x):
        for layer in self.layers_list:
            x = x + layer(x)
        return x


class Encoder(Model):
    """
    Encoder network that uses convolutional layers to process input sequences.
    Contains optional normalization, dilation, and GLU layers.
    """
    def __init__(
        self,
        out_channels,
        kernel_size=8,
        stride=4,
        norm_groups=1,
        empty=False,
        freq=True,
        dconv=True,
        norm=True,
        context=0,
        dconv_kw={},
        pad=True,
        rewrite=True,
    ):

        super().__init__()
        norm_fn = lambda: IdentityNorm()
       
        if norm:
            norm_fn = lambda: GroupNormalization(norm_groups,  epsilon = 0.00001)
        if pad:
            padding = "same"
        else:
            padding = "valid"
        self.freq = freq
        self.kernel_size = kernel_size
        self.stride = stride
        self.empty = empty
        self.norm = norm
        self.glu = GLU(axis=-1)  # halves the number of channels
        Conv = Conv1D
        if freq:
            kernel_size = [kernel_size, 1]
            stride = [stride, 1]
            Conv = Conv2D

        self.conv = Conv(out_channels, kernel_size, stride, padding)
        if self.empty:
            return

        self.norm1 = norm_fn()
        self.rewrite = None
        if rewrite:
            self.rewrite = Conv(2 * out_channels, 1 + 2 * context, 1)
            self.norm2 = norm_fn()

        self.dconv = None
        if dconv:
            self.dconv = DConv(out_channels, **dconv_kw)

    def call(self, x):
        # channel last
        if not self.freq:
            le = x.shape[1]
            if not le % self.stride == 0:
                pad_amount = self.stride - (le % self.stride)
                paddings = [[0, 0], [0, pad_amount], [0, 0]]
                x = tf.pad(x, paddings)

        y = self.conv(x)

        if self.empty:
            return y
        y = activations.gelu(self.norm1(y))
        if self.dconv:
            tensor_shape = tf.shape(y)
            if self.freq:
                B, Frame, Freq, C = y.shape
                y = tf.reshape(y, (-1, Freq, C))

            y = self.dconv(y)

            if self.freq:
                y = tf.reshape(y, (tensor_shape[0], Frame, Freq, C))

        if self.rewrite:
            y = self.rewrite(y)
            z = self.norm2(y)
            z = self.glu(z)
        else:
            z = y

        return z


class Decoder(Model):
    """
    Decoder network that uses transposed convolutional layers to upsample input sequences.
    Contains optional normalization, dilation, and GLU layers.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        last=False,
        kernel_size=8,
        stride=4,
        norm_groups=1,
        empty=False,
        freq=True,
        dconv=True,
        norm=True,
        context=1,
        dconv_kw={},
        pad=True,
        context_freq=True,
        rewrite=True,
    ):

        super().__init__()
        norm_fn = lambda: IdentityNorm()
        if norm:
            norm_fn = lambda: GroupNormalization(norm_groups)
        if pad:
            padding = "same"
        else:
            padding = "valid"

        self.units = out_channels
        self.last = last
        self.freq = freq
        self.empty = empty
        self.stride = stride
        self.kernel_size = kernel_size
        self.norm = norm
        self.glu = GLU(axis=-1)  # halves the number of channels
        self.context_freq = context_freq
        Conv = Conv1D
        ConvTr = Conv1DTranspose

        if freq:
            kernel_size = [kernel_size, 1]
            stride = [stride, 1]
            Conv = Conv2D
            ConvTr = Conv2DTranspose

        self.conv_tr = ConvTr(out_channels, kernel_size, stride, padding)
        self.norm2 = norm_fn()

        if self.empty:
            return

        self.rewrite = None
        if rewrite:
            self.rewrite = Conv(2 * in_channels, 1 + 2 * context, 1,padding)
            self.norm1 = norm_fn()

        self.dconv = None
        if dconv:
            self.dconv = DConv(out_channels, **dconv_kw)

    def build(self, input_shape):
        self.built = True  # Important to set this flag to True

    def call(self, x, skip=None):
        if not self.empty:
            x = x + skip
            if self.rewrite:
                y = self.glu(self.norm1(self.rewrite(x)))
            else:
                y = x
            if self.dconv:
                if self.freq:
                    B, Frame, Freq, C = y.shape
                    y = tf.reshape(y, (-1, Freq, C))
                y = self.dconv(y)
                if self.freq:
                    y = tf.reshape(y, (B, Frame, Freq, C))

        else:
            y = x
            assert skip is None, "empty layer should not have skip"

        z = self.norm2(self.conv_tr(y))

        if not self.last:
            z = activations.gelu(z)

        return z


# **************************************** Test model ***************************************


if __name__ == "__main__":
    from stft_utils import STFTUtils

    class TestHTDemucs(Model):
        """This is a test model.
        Not to be used in practice."""

        def __init__(self, **kwargs):
            super().__init__()
            self.z_encoder = []
            self.z_decoder = []
            self.t_encoder = []
            self.t_decoder = []

            enc = [48, 96, 192, 384]
            dec = [192, 96, 48, 1]

            for ef, df in zip(enc, dec):
                self.t_encoder.append(Encoder(ef, freq=False, **kwargs))
                self.t_decoder.append(Decoder(df, freq=False, empty=True, **kwargs))
                self.z_encoder.append(Encoder(ef, **kwargs))
                self.z_decoder.append(Decoder(df, empty=True, **kwargs))

        def build(self, input_shape):
            self.built = True

        def call(self, x):
            y = tf.transpose(x, [0, 2, 1])
            y = STFTUtils.spectro(y)
            z = STFTUtils.magnitude(y)
            z = tf.transpose(z, [0, 3, 2, 1])

            for t_encode, z_encode in zip(self.t_encoder, self.z_encoder):
                x = t_encode(x)
                z = z_encode(z)
                print("T_encode", x.shape, "Z_encode", z.shape)

            print("--------------------------------------------------")

            for t_decode, z_decode in zip(self.t_decoder, self.z_decoder):
                x = t_decode(x)
                z = z_decode(z)
                print("T_decode", x.shape, "Z_decode", z.shape)

            return x, z

    # Define the input shape and sample input
    batch_size = 1
    seq_len = 160000
    channels = 1
    input_data = np.random.randn(batch_size, seq_len, channels).astype(
        np.float16
    )  # For TensorFlow, shape (batch_size, seq_len, channels)

    # TensorFlow Model Test instantiation
    model = TestHTDemucs()
    input_tf = tf.convert_to_tensor(input_data)
    output_tf = model(input_tf)

