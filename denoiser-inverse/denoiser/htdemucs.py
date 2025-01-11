import tensorflow as tf
from .stft_utils import STFTUtils
from .hybrid_encoder_decoder import Encoder, Decoder
from .transformerLayers.transformers import CrossDomainTransformerEncoder


# will require to update this for training
class ScaledEmbedding(tf.keras.layers.Layer):
    """
    Boost learning rate for embeddings (with scale).
    Also, can make embeddings continuous with smooth.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, scale: float = 10., smooth=False):
        super(ScaledEmbedding, self).__init__()
        self.embedding = tf.keras.layers.Embedding(num_embeddings, embedding_dim)
        self.scale = scale
        self.smooth = smooth
        self.built = False

    def build(self, input_shape):
        super(ScaledEmbedding, self).build(input_shape)
        if not self.built:
            if self.smooth:
                weight = tf.cumsum(self.embedding.embeddings, axis=0)
                weight = weight / tf.math.sqrt(tf.range(1, tf.shape(weight)[0] + 1, dtype=tf.float32))[:, None]
                self.embedding.embeddings.assign(weight)
            self.embedding.embeddings.assign(self.embedding.embeddings / self.scale)
            self.built = True

    @property
    def weight(self):
        return self.embedding.embeddings * self.scale

    def call(self, inputs):
        if not self.built:
            self.build(None)
        out = self.embedding(inputs) * self.scale
        return out


@tf.keras.utils.register_keras_serializable()
class HTDemucs(tf.keras.Model):
    """
    Spectrogram and hybrid Demucs model.
    The spectrogram model has the same structure as Demucs, except the first few layers are over the
    frequency axis, until there is only 1 frequency, and then it moves to time convolutions.
    Frequency layers can still access information across time steps thanks to the DConv residual.

    Hybrid model have a parallel time branch. At some layer, the time branch has the same stride
    as the frequency branch and then the two are combined. The opposite happens in the decoder.

    Models can either use naive iSTFT from masking, Wiener filtering ([Ulhih et al. 2017]),
    or complex as channels (CaC) [Choi et al. 2020]. Wiener filtering is based on
    Open Unmix implementation [Stoter et al. 2019].

    The loss is always on the temporal domain, by backpropagating through the above
    output methods and iSTFT. This allows to define hybrid models nicely. However, this breaks
    a bit Wiener filtering, as doing more iteration at test time will change the spectrogram
    contribution, without changing the one from the waveform, which will lead to worse performance.
    I tried using the residual option in OpenUnmix Wiener implementation, but it didn't improve.
    CaC on the other hand provides similar performance for hybrid, and works naturally with
    hybrid models.

    This model also uses frequency embeddings are used to improve efficiency on convolutions
    over the freq. axis, following [Isik et al. 2020] (https://arxiv.org/pdf/2008.04470.pdf).

    Unlike classic Demucs, there is no resampling here, and normalization is always applied.
    """

    def __init__(
        self,
        # Channels
        audio_channels=1,
        channels=48,
        channels_time=None,
        growth=2,
        # STFT
        nfft=4096,
        wiener_iters=0,
        end_iters=0,
        wiener_residual=False,
        cac=True,
        # Main structure
        depth=4,
        rewrite=True,
        # Frequency branch
        multi_freqs=None,
        multi_freqs_depth=3,
        freq_emb=0.2,
        emb_scale=10,
        emb_smooth=True,
        # Convolutions
        kernel_size=8,
        time_stride=2,
        stride=4,
        context=1,
        context_enc=0,
        # Normalization
        norm_starts=4,
        norm_groups=4,
        # DConv residual branch
        dconv_mode=1,
        dconv_depth=2,
        dconv_comp=8,
        dconv_init=1e-3,
        # Transformer
        t_layers=5,
        t_emb="sin",
        t_hidden_scale=4.0,
        t_heads=8,
        t_dropout=0.0,
        t_max_positions=10000,
        t_norm_in=True,
        t_norm_in_group=False,
        t_group_norm=False,
        t_norm_first=True,
        t_norm_out=True,
        t_max_period=10000.0,
        t_weight_decay=0.0,
        t_lr=None,
        t_layer_scale=True,
        t_gelu="gelu",
        t_weight_pos_embed=1.0,
        t_sin_random_shift=0,
        t_cape_mean_normalize=True,
        t_cape_augment=True,
        t_cape_glob_loc_scale=[5000.0, 1.0, 1.4],
        t_sparse_self_attn=False,
        t_sparse_cross_attn=False,
        t_mask_type="diag",
        t_mask_random_seed=42,
        t_sparse_attn_window=500,
        t_global_window=100,
        t_sparsity=0.95,
        t_auto_sparsity=False,
        # ------ Particuliar parameters
        t_cross_first=False,
        # Weight init
        rescale=0.1,
        # Metadata
        sample_rate=16000,
        length=10,
        use_train_segment=True,
        **kwargs,
    ):
        """
        Args:
            audio_channels (int): input/output audio channels.
            channels (int): initial number of hidden channels.
            channels_time: if not None, use a different `channels` value for the time branch.
            growth: increase the number of hidden channels by this factor at each layer.
            nfft: number of fft bins. Note that changing this require careful computation of
                various shape parameters and will not work out of the box for hybrid models.
            wiener_iters: when using Wiener filtering, number of iterations at test time.
            end_iters: same but at train time. For a hybrid model, must be equal to `wiener_iters`.
            wiener_residual: add residual source before wiener filtering.
            cac: uses complex as channels, i.e. complex numbers are 2 channels each
                in input and output. no further processing is done before ISTFT.
            depth (int): number of layers in the encoder and in the decoder.
            rewrite (bool): add 1x1 convolution to each layer.
            multi_freqs: list of frequency ratios for splitting frequency bands with `MultiWrap`.
            multi_freqs_depth: how many layers to wrap with `MultiWrap`. Only the outermost
                layers will be wrapped.
            freq_emb: add frequency embedding after the first frequency layer if > 0,
                the actual value controls the weight of the embedding.
            emb_scale: equivalent to scaling the embedding learning rate
            emb_smooth: initialize the embedding with a smooth one (with respect to frequencies).
            kernel_size: kernel_size for encoder and decoder layers.
            stride: stride for encoder and decoder layers.
            time_stride: stride for the final time layer, after the merge.
            context: context for 1x1 conv in the decoder.
            context_enc: context for 1x1 conv in the encoder.
            norm_starts: layer at which group norm starts being used.
                decoder layers are numbered in reverse order.
            norm_groups: number of groups for group norm.
            dconv_mode: if 1: dconv in encoder only, 2: decoder only, 3: both.
            dconv_depth: depth of residual DConv branch.
            dconv_comp: compression of DConv branch.
            dconv_attn: adds attention layers in DConv branch starting at this layer.
            dconv_lstm: adds a LSTM layer in DConv branch starting at this layer.
            dconv_init: initial scale for the DConv branch LayerScale.
            bottom_channels: if >0 it adds a linear layer (1x1 Conv) before and after the
                transformer in order to change the number of channels
            t_layers: number of layers in each branch (waveform and spec) of the transformer
            t_emb: "sin", "cape" or "scaled"
            t_hidden_scale: the hidden scale of the Feedforward parts of the transformer
                for instance if C = 384 (the number of channels in the transformer) and
                t_hidden_scale = 4.0 then the intermediate layer of the FFN has dimension
                384 * 4 = 1536
            t_heads: number of heads for the transformer
            t_dropout: dropout in the transformer
            t_max_positions: max_positions for the "scaled" positional embedding, only
                useful if t_emb="scaled"
            t_norm_in: (bool) norm before addinf positional embedding and getting into the
                transformer layers
            t_norm_in_group: (bool) if True while t_norm_in=True, the norm is on all the
                timesteps (GroupNorm with group=1)
            t_group_norm: (bool) if True, the norms of the Encoder Layers are on all the
                timesteps (GroupNorm with group=1)
            t_norm_first: (bool) if True the norm is before the attention and before the FFN
            t_norm_out: (bool) if True, there is a GroupNorm (group=1) at the end of each layer
            t_max_period: (float) denominator in the sinusoidal embedding expression
            t_weight_decay: (float) weight decay for the transformer
            t_lr: (float) specific learning rate for the transformer
            t_layer_scale: (bool) Layer Scale for the transformer
            t_gelu: (bool) activations of the transformer are GeLU if True, ReLU else
            t_weight_pos_embed: (float) weighting of the positional embedding
            t_cape_mean_normalize: (bool) if t_emb="cape", normalisation of positional embeddings
                see: https://arxiv.org/abs/2106.03143
            t_cape_augment: (bool) if t_emb="cape", must be True during training and False
                during the inference, see: https://arxiv.org/abs/2106.03143
            t_cape_glob_loc_scale: (list of 3 floats) if t_emb="cape", CAPE parameters
                see: https://arxiv.org/abs/2106.03143
            t_sparse_self_attn: (bool) if True, the self attentions are sparse
            t_sparse_cross_attn: (bool) if True, the cross-attentions are sparse (don't use it
                unless you designed really specific masks)
            t_mask_type: (str) can be "diag", "jmask", "random", "global" or any combination
                with '_' between: i.e. "diag_jmask_random" (note that this is permutation
                invariant i.e. "diag_jmask_random" is equivalent to "jmask_random_diag")
            t_mask_random_seed: (int) if "random" is in t_mask_type, controls the seed
                that generated the random part of the mask
            t_sparse_attn_window: (int) if "diag" is in t_mask_type, for a query (i), and
                a key (j), the mask is True id |i-j|<=t_sparse_attn_window
            t_global_window: (int) if "global" is in t_mask_type, mask[:t_global_window, :]
                and mask[:, :t_global_window] will be True
            t_sparsity: (float) if "random" is in t_mask_type, t_sparsity is the sparsity
                level of the random part of the mask.
            t_cross_first: (bool) if True cross attention is the first layer of the
                transformer (False seems to be better)
            rescale: weight rescaling trick
            use_train_segment: (bool) if True, the actual size that is used during the
                training is used during inference.
        """
        super().__init__()
        self._init_kwargs = {
            "audio_channels": audio_channels,
            "channels": channels,
            "channels_time": channels_time,
            "growth": growth,
            "nfft": nfft,
            "wiener_iters": wiener_iters,
            "end_iters": end_iters,
            "wiener_residual": wiener_residual,
            "cac": cac,
            "depth": depth,
            "rewrite": rewrite,
            "multi_freqs": multi_freqs,
            "multi_freqs_depth": multi_freqs_depth,
            "freq_emb": freq_emb,
            "emb_scale": emb_scale,
            "emb_smooth": emb_smooth,
            "kernel_size": kernel_size,
            "time_stride": time_stride,
            "stride": stride,
            "context": context,
            "context_enc": context_enc,
            "norm_starts": norm_starts,
            "norm_groups": norm_groups,
            "dconv_mode": dconv_mode,
            "dconv_depth": dconv_depth,
            "dconv_comp": dconv_comp,
            "dconv_init": dconv_init,
            "t_layers": t_layers,
            "t_emb": t_emb,
            "t_hidden_scale": t_hidden_scale,
            "t_heads": t_heads,
            "t_dropout": t_dropout,
            "t_max_positions": t_max_positions,
            "t_norm_in": t_norm_in,
            "t_norm_in_group": t_norm_in_group,
            "t_group_norm": t_group_norm,
            "t_norm_first": t_norm_first,
            "t_norm_out": t_norm_out,
            "t_max_period": t_max_period,
            "t_weight_decay": t_weight_decay,
            "t_lr": t_lr,
            "t_layer_scale": t_layer_scale,
            "t_gelu": t_gelu,
            "t_weight_pos_embed": t_weight_pos_embed,
            "t_sin_random_shift": t_sin_random_shift,
            "t_cape_mean_normalize": t_cape_mean_normalize,
            "t_cape_augment": t_cape_augment,
            "t_cape_glob_loc_scale": t_cape_glob_loc_scale,
            "t_sparse_self_attn": t_sparse_self_attn,
            "t_sparse_cross_attn": t_sparse_cross_attn,
            "t_mask_type": t_mask_type,
            "t_mask_random_seed": t_mask_random_seed,
            "t_sparse_attn_window": t_sparse_attn_window,
            "t_global_window": t_global_window,
            "t_sparsity": t_sparsity,
            "t_auto_sparsity": t_auto_sparsity,
            "t_cross_first": t_cross_first,
            "rescale": rescale,
            "sample_rate": sample_rate,
            "length": length,
            "use_train_segment": use_train_segment,
        }
        self._init_kwargs.update(kwargs)
        self.cac = cac
        self.wiener_residual = wiener_residual
        self.sources = ["voice"]
        self.audio_channels = audio_channels
        self.kernel_size = kernel_size
        self.context = context
        self.stride = stride
        self.depth = depth
        self.channels = channels
        self.sample_rate = sample_rate
        self.length = length
        self.use_train_segment = use_train_segment
        self.nfft = nfft
        self.hop_length = nfft // 4
        self.wiener_iters = wiener_iters
        self.end_iters = end_iters
        self.freq_emb = freq_emb
        assert wiener_iters == end_iters

        self.freq_encoder = []
        self.freq_decoder = []

        self.tencoder = []
        self.tdecoder = []

        chin = audio_channels
        chin_z = chin  # number of channels for the freq branch
        if self.cac:
            chin_z *= 2
        chout = channels_time or channels
        chout_z = channels
        freqs = nfft // 2

        for index in range(depth):
            norm = index >= norm_starts
            freq = freqs > 1
            ker = kernel_size
            stri = stride
            if not freq:
                assert freqs == 1
                ker = time_stride * 2
                stri = time_stride
            if index == 0:
                # for fs 32k stri = 2, fs 48k stri = 3
                stri = self.sample_rate // 16000

            pad = True
            last_freq = False
            if freq and freqs <= kernel_size:
                ker = freqs
                pad = False
                last_freq = True

            kw = {
                "kernel_size": ker,
                "stride": stri,
                "freq": freq,
                "pad": pad,
                "norm": norm,
                "rewrite": rewrite,
                "norm_groups": norm_groups,
                "dconv_kw": {
                    "depth": dconv_depth,
                    "compress": dconv_comp,
                    "init": dconv_init,
                    "gelu": True,
                },
            }
            kwt = dict(kw)
            kwt["freq"] = 0
            kwt["kernel_size"] = kernel_size
            kwt["stride"] = stri
            kwt["pad"] = True
            enc = Encoder(chout_z, dconv=dconv_mode & 1, **kw)
            if freq:
                tenc = Encoder(chout, dconv=dconv_mode & 1, **kwt)
                self.tencoder.append(tenc)

            self.freq_encoder.append(enc)
            if index == 0:
                chin = self.audio_channels * len(self.sources)
                chin_z = chin
                if self.cac:
                    chin_z *= 2
            dec = Decoder(
                chout_z,
                chin_z,
                dconv=dconv_mode & 2,
                last=index == 0,
                context=context,
                **kw,
            )
            if freq:
                tdec = Decoder(
                    chout_z,
                    chin,
                    dconv=dconv_mode & 2,
                    last=index == 0,
                    context=context,
                    **kwt,
                )
                self.tdecoder.insert(0, tdec)
            self.freq_decoder.insert(0, dec)

            chin = chout
            chin_z = chout_z
            chout = int(growth * chout)
            chout_z = int(growth * chout_z)

            if freq:
                if freqs <= kernel_size:
                    freqs = 1
                else:
                    freqs //= stri
            if index == 0 and freq_emb:
                self.freq_emb = ScaledEmbedding(
                    freqs, chin_z, scale=emb_scale, smooth=emb_smooth
                )
                self.freq_emb_scale = freq_emb

        transformer_channels = channels * growth ** (depth - 1)
        if t_layers > 0:
            kw_transformers = {
                "dim_embed": transformer_channels,
                "hidden_scale": t_hidden_scale,
                "num_attention_heads": t_heads,
                "num_layers": t_layers,
                "cross_attention_first": t_cross_first,
                "dropout": t_dropout,
                "normalize_input": t_norm_in,
                "pre_norm": t_norm_first,
                "post_norm": t_norm_out,
                "use_layer_scale": t_layer_scale,
                "activation": t_gelu,
                "max_period": t_max_period,
                "weight_pos_embed": t_weight_pos_embed,
                "sin_random_shift": t_sin_random_shift,
            }

            self.cross_domain_transformer = CrossDomainTransformerEncoder(
                **kw_transformers
            )

    def _mask(self, x):
        if self.cac:
            real = x[..., 0]
            imag = x[..., 1]
            complex_tensor = tf.complex(real, imag)
            return complex_tensor

    def call(self, mix, training=None):  # Batch, timestamps, channel
        signal_length = mix.shape[1]
        xt = mix
        mix = tf.transpose(mix, [0, 2, 1])
        z = STFTUtils.spectro(mix, self.nfft)  # batch, channel, frames, frequency
        mag = STFTUtils.magnitude(z)  # batch, channel, frames, frequency
        x_freq = mag  # x_freq is frequency domain tensor

        # Normalize frequency and time branches
        mean_freq = tf.math.reduce_mean(x_freq, axis=(1, 2, 3), keepdims=True)
        std_freq = tf.math.reduce_std(x_freq, axis=(1, 2, 3), keepdims=True)
        x_freq = (x_freq - mean_freq) / (1e-5 + std_freq)

        meant = tf.math.reduce_mean(xt, axis=(1, 2), keepdims=True)
        stdt = tf.math.reduce_std(xt, axis=(1, 2), keepdims=True)
        xt = (xt - meant) / (1e-5 + stdt)

        x_freq = tf.transpose(x_freq, [0, 3, 2, 1])  # batch, frequency, frames, channel

        saved_freq = []  # skip connections, freq.
        saved_t = []  # skip connections, time.
        lengths_freq = []  # saved lengths to properly remove padding, freq branch.
        lengths_t = []  # saved lengths for time branch.

        # Encoder layer
        for idx, freq_encode in enumerate(self.freq_encoder):
            lengths_freq.append(x_freq.shape[-2])  # Frame length
            if idx < len(self.tencoder):
                # we have not yet merged branches.
                lengths_t.append(xt.shape[1])  # timestamp dimension length
                tenc = self.tencoder[idx]
                xt = tenc(xt)
                saved_t.append(xt)  # save for skip connection

            x_freq = freq_encode(x_freq)  # batch, frequency, frames, channel
            if idx == 0 and self.freq_emb is not None:
                frequencies = tf.range(tf.shape(x_freq)[1])
                emb = self.freq_emb(frequencies)
                emb = emb[None, :, None, :]
                emb = tf.broadcast_to(emb, tf.shape(x_freq))
                embedding_amount = self.freq_emb_scale * emb
                x_freq = x_freq + embedding_amount
            saved_freq.append(x_freq)

        x_freq = tf.transpose(x_freq, [0, 3, 1, 2])  # batch, channel, frequncy, frame
        xt = tf.transpose(xt, [0, 2, 1])  # batch, channel, timasteps

        # Apply cross-domain transformer if needed
        if self.cross_domain_transformer:
            x_freq, xt = self.cross_domain_transformer(x_freq, xt)

        x_freq = tf.transpose(x_freq, [0, 2, 3, 1])  # batch, frequency, frames, channel
        xt = tf.transpose(xt, [0, 2, 1])  # batch, timasteps, channel

        # Decoder
        for idx, freq_decode in enumerate(self.freq_decoder):
            skip_freq = saved_freq.pop(-1)
            skip_time = saved_t.pop(-1)
            x_freq = freq_decode(x_freq, skip_freq)
            xt = self.tdecoder[idx](xt, skip_time)

        # Denormalize frequency domain output
        x_freq = x_freq * std_freq + mean_freq
        x_freq = self._mask(x_freq)  # batch, frequency, frames
        x_freq = tf.transpose(x_freq, [0, 2, 1])  # batch, frames, frequency

        x_freq_shape = tf.shape(x_freq)
        x_freq = tf.reshape(
            x_freq, [x_freq_shape[0], 1, x_freq.shape[1], x_freq.shape[2]]
        )  # batch, channel, frame, frequency
        x_freq = STFTUtils.inverse_spectro(
            x_freq, signal_length=signal_length
        )  # batch, frequency, frames
        x_freq = tf.transpose(x_freq, [0, 2, 1])  # batch, signal, channel

        # Denormalize time domain output
        xt = xt * stdt + meant

        final_output = x_freq + xt

        return final_output

    def build(self, input_shape):
        self.call(tf.zeros(input_shape))  # build the model with all weights and biases
        self.built = True
        print("Model built")

    def get_config(self):
        return self._init_kwargs

    def _get_save_spec(self, **kwargs):
        return tf.TensorSpec([None, self.sample_rate * self.length, 1], tf.float32)
