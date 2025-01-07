import math
from typing import Tuple
import tensorflow as tf


class STFTUtils:
    """Utility class for performing Short-Time Fourier Transform (STFT) operations."""

    @staticmethod
    @tf.function
    def pad_1d(
        tensor: tf.Tensor,
        paddings: Tuple[int, int],
        mode: str = "CONSTANT",
        value: float = 0.0,
    ) -> tf.Tensor:
        """
        Pads a 1D tensor with the specified padding values and mode.

        This function wraps around `tf.pad`, specifically handling reflect padding for
        small input tensors by adding extra padding if necessary as tf.pad cannot apply
        padding when input tensor is smaller than padding.

        Args:
            tensor (tf.Tensor): The input tensor to pad.
            paddings (Tuple[int, int]): A tuple of two integers specifying the padding
                    to add to the left and right of the tensor.
            mode (str): Padding mode, can be "CONSTANT", "REFLECT", or any other mode
                    supported by `tf.pad`. Default is "CONSTANT".

        Returns:
            tf.Tensor: The padded tensor.
        """
        tensor_copy = tensor
        length = tensor.shape[-1]
        padding_left, padding_right = paddings

        if mode == "REFLECT":
            max_padding = max(padding_left, padding_right)
            if length <= max_padding:
                extra_padding = max_padding - length + 1
                extra_padding_right = min(padding_right, extra_padding)
                extra_padding_left = extra_padding - extra_padding_right

                paddings = (
                    padding_left - extra_padding_left,
                    padding_right - extra_padding_right,
                )

                # zero padding on both side
                tensor = tf.pad(
                    tensor,
                    [[0, 0], [0, 0], [extra_padding_left, extra_padding_right]],
                    mode="CONSTANT",
                    constant_values=value,
                )

        # Apply the main padding
        padded_tensor = tf.pad(
            tensor,
            [[0, 0], [0, 0], paddings],
            mode=mode,
            constant_values=value,  # Ignored when mode = "REFLECT"
        )

        # Ensure the shape and values
        assert padded_tensor.shape[-1] == length + padding_left + padding_right
        tf.debugging.assert_equal(
            padded_tensor[..., padding_left : padding_left + length],
            tensor_copy,
        )

        return padded_tensor

    @staticmethod
    @tf.function
    def spectro(
        signal: tf.Tensor, n_fft: int = 4096, hop_length: int = None
    ) -> tf.Tensor:
        """
        Computes the Short-Time Fourier Transform (STFT) of the input tensor.

        This function computes the spectrogram of the input tensor using STFT.

        Args:
            signal (tf.Tensor): The input tensor for which to compute the spectrogram.
                    Expected to be of shape (``batch``,``channel``, ``signal_length``).
            n_fft (int): The size of the FFT to apply. Default is 4096.
            hop_length (int): The hop length (number of samples between successive frames).
                    If None, defaults to `n_fft // 4`. Default is None.

        Returns:
            tf.Tensor: A tensor containing the STFT result with shape (batch, channel, frames, frequency)
        """

        if hop_length is None:
            hop_length = n_fft // 4

        assert hop_length == n_fft // 4, "hop_length should be n_fft // 4"

        frames = int(math.ceil(signal.shape[-1] / hop_length))

        pad_left = hop_length // 2 * 3
        pad_right = pad_left + frames * hop_length - signal.shape[-1]
        pad_second = n_fft // 2

        # shape (batch, channel, signal) -> (batch, channel, signal + pad_left + pad_right)
        padded_signal = STFTUtils.pad_1d(signal, (pad_left, pad_right), "REFLECT")


        padded_signal = STFTUtils.pad_1d(padded_signal, (pad_second, pad_second), "REFLECT")  
        window_fn = tf.signal.hann_window
        padded_signal_shape = tf.shape(padded_signal)
        padded_signal = tf.reshape(padded_signal, [padded_signal_shape[0]*padded_signal_shape[1], padded_signal_shape[2]])
        stfts = tf.signal.stft(
            padded_signal,
            frame_length=n_fft,
            frame_step=hop_length,
            fft_length=n_fft,
            window_fn=window_fn,
        )  # shape (batch, channel, frames, frequency)

        _, frames_sec, frequency = stfts.shape
        stfts = tf.reshape(stfts, [padded_signal_shape[0], padded_signal_shape[1], frames_sec, frequency])
        # Do Normalization
        normalization_factor = tf.cast(n_fft, tf.float32) ** -0.5  # Normalization factor         
        stfts = stfts * tf.cast(normalization_factor, stfts.dtype)    # Apply the normalization

        # Remove extra dimension of frames and frequency
        # shape (batch, channel, fft_bins, fft_length // 2 + 1) -> (batch, channel, fft_bins - 4, fft_length // 2)
        stfts = stfts[..., 2 : 2 + frames, :-1]
        return stfts  # shape (batch, channel, fft_bins - 4, fft_length // 2)

    @staticmethod
    @tf.function
    def inverse_spectro(spectrograms, hop_length=None, signal_length=None) -> tf.Tensor:
        """
        Computes the inverse Short-Time Fourier Transform (iSTFT) of the input tensor.

        This function performs the inverse STFT operation on the input tensor to
        reconstruct the original signal.

        Args:
            spectrograms (tf.Tensor): The input tensor. Expected to be in the format of an STFT output.
                    This is a complex tensor of shape (``batch``, ``channel``, ``n_frame``, ``fft_size``),
                    where the ``channel`` dimension is optional.
            hop_length (int): The hop length (number of samples between successive frames).
                    If None, defaults to `n_fft // 4`. Default is None.
            signal_length (int): The desired length of the reconstructed signal. If not provided,
                    the length is determined based on the input tensor shape.

        Returns:
            tf.Tensor: The reconstructed tensor after applying inverse STFT. The shape is
                    (``batch``, ``channel``, ``signal_length``).
        """
        # ================= padding section =================
        # Frames pad (last dimension) Frequency pad (last 2nd dimension)
        # shape (batch, channel, frames, frequency) -> (batch, channel, frames, frequency)

        ##print("inverse spectro ", spectrograms ,tf.reduce_sum(tf.square(spectrograms)))
        spectrograms = tf.pad(spectrograms, [[0, 0], [0, 0], [2, 2], [0, 1]])
        # ===================================================
        ##print("inverse spectro ", spectrograms ,tf.reduce_sum(tf.square(spectrograms)))

        *other_dims, num_freqs = spectrograms.shape
        n_fft = 2 * num_freqs - 2

        if hop_length is None:
            hop_length = n_fft // 4
        window_fn = tf.signal.hann_window
        
        normalization_factor = tf.cast(n_fft, tf.float32) ** -0.5  

        # window = tf.signal.hann_window(n_fft)
        # normalization_factor = tf.reduce_sum(window ** 2) ** 0.5
        spectrograms = spectrograms / tf.cast(normalization_factor, spectrograms.dtype) 

        signal = tf.signal.inverse_stft(
            spectrograms,
            frame_length=n_fft,
            frame_step=hop_length,
            fft_length=n_fft,
            window_fn=tf.signal.inverse_stft_window_fn(
             hop_length, forward_window_fn=window_fn)
            )  # shape (batch, channel, signal_length + padding)
        
        padding = (hop_length // 2 * 3) + (n_fft // 2)  # Amount of padding that was applied

        
        signal = signal[..., padding : signal_length + padding]

        # print("output inverse spectro ", signal ,tf.reduce_sum(tf.square(signal)))

        return signal  # shape (batch, channel, signal_length)

    @staticmethod
    @tf.function
    def magnitude(spectrogram: tf.Tensor, complex_as_channel: bool = True) -> tf.Tensor:
        """
        Returns the magnitude of the spectrogram.

        If `complex_as_channel` is True, moves the complex dimension to the channel one,
        thereby converting the complex spectrogram into a tensor with two
        channels representing the real and imaginary parts.

        Args:
            spectrogram (tf.Tensor): The input tensor representing the spectrogram.
                    Expected shape is (``batch``, ``channel``, ``fft_bins``, ``fft_size``).
            complex_as_channel (bool): If True, converts the complex spectrogram into a tensor
                    with channels for real and imaginary parts. Default is False.

        Returns:
            tf.Tensor: The magnitude of the spectrogram, or the real and imaginary
                    parts as separate channels if `complex_as_channel` is True. The shape
                    is (``batch``, ``channel``, ``fft_bins``, ``fft_size``).
        """
        if complex_as_channel:
            batch, channels, frames, freq_bins = spectrogram.shape
            spectrogram_shape = tf.shape(spectrogram)

            # Convert to real view and permute dimensions
            real = tf.math.real(spectrogram)
            imaginary = tf.math.imag(spectrogram)

            combined = tf.concat(
                [real[..., tf.newaxis], imaginary[..., tf.newaxis]],
                axis=-1,
            )
            combined = tf.transpose(combined, perm=[0, 1, 4, 2, 3])

            magnitude = tf.reshape(
                combined, (spectrogram_shape[0], channels * 2, frames, freq_bins)
            )  # shape (batch, 2 * channel, fft_bins, fft_size)
        else:
            magnitude = tf.abs(
                spectrogram
            )  # shape (batch, channel, fft_bins, fft_size)

        return magnitude
