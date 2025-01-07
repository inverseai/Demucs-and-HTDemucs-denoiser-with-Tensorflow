from tensorflow.keras.layers import (
    MultiHeadAttention,
    Dense,
    Dropout,
    LayerNormalization,
    GroupNormalization,
)
import tensorflow as tf
import numpy as np
import einops

tf.random.set_seed(0)
# tf.config.run_functions_eagerly(True)  # Enable eager mode globally


class LayerScale(tf.keras.layers.Layer):
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

    def call(self, inputs):
        if self.channel_last:
            return self.scale * inputs
        else:
            # Reshape scale for broadcasting
            return tf.reshape(self.scale, [-1, 1]) * inputs


@tf.function
def create_sin_embedding(length, dim, shift=0, max_period=10000.0):
    # assert dim % 2 == 0

    pos = shift + tf.range(length, dtype=tf.float32)
    pos = tf.reshape(pos, (-1, 1, 1))

    half_dim = dim // 2
    adim = tf.range(half_dim, dtype=tf.float32)
    adim = tf.reshape(adim, (1, 1, -1))
    phase = pos / (max_period ** (adim / tf.cast((half_dim - 1), tf.float32)))

    return tf.concat(
        [
            tf.cos(phase),
            tf.sin(phase),
        ],
        axis=-1,
    )


@tf.function
def create_2d_sin_embedding(dim_emb, frequency, frames, max_period=10000):

    # if dim_emb % 4 != 0:
    #     raise ValueError("Dimension must be a multiple of 4.")

    # Initialize the position encoding tensor with zeros
    pos_enc = tf.zeros((dim_emb, frequency, frames), dtype=tf.float32)
    dim_emb_half = tf.cast(dim_emb / 2, tf.float32)

    div_term = tf.exp(
        tf.range(0, dim_emb_half, 2, dtype=tf.float32)
        * -(tf.math.log(max_period) / dim_emb_half)
    )
    pos_w = tf.range(0, frames, dtype=tf.float32)[:, tf.newaxis]
    pos_h = tf.range(0, frequency, dtype=tf.float32)[:, tf.newaxis]

    sin_w = tf.repeat(
        tf.expand_dims(tf.transpose(tf.sin(pos_w * div_term), perm=[1, 0]), axis=1),
        repeats=[frequency],
        axis=1,
    )
    cos_w = tf.repeat(
        tf.expand_dims(tf.transpose(tf.cos(pos_w * div_term), perm=[1, 0]), axis=1),
        repeats=[frequency],
        axis=1,
    )

    sin_h = tf.repeat(
        tf.expand_dims(tf.transpose(tf.sin(pos_h * div_term), perm=[1, 0]), axis=2),
        repeats=[frames],
        axis=2,
    )
    cos_h = tf.repeat(
        tf.expand_dims(tf.transpose(tf.cos(pos_h * div_term), perm=[1, 0]), axis=2),
        repeats=[frames],
        axis=2,
    )

    dim_emb_half = tf.cast(dim_emb_half, tf.int32)

    # Create indices for the updates
    indices_w = tf.range(0, dim_emb_half, 2)
    indices_w = tf.expand_dims(indices_w, axis=1)

    indices_w_plus_1 = tf.range(1, dim_emb_half, 2)
    indices_w_plus_1 = tf.expand_dims(indices_w_plus_1, axis=1)

    indices_h = tf.range(dim_emb_half, 2 * dim_emb_half, 2)
    indices_h = tf.expand_dims(indices_h, axis=1)

    indices_h_plus_1 = tf.range(dim_emb_half + 1, 2 * dim_emb_half, 2)
    indices_h_plus_1 = tf.expand_dims(indices_h_plus_1, axis=1)

    # Scatter updates
    pos_enc = tf.tensor_scatter_nd_update(pos_enc, indices_w, sin_w)
    pos_enc = tf.tensor_scatter_nd_update(pos_enc, indices_w_plus_1, cos_w)
    pos_enc = tf.tensor_scatter_nd_update(pos_enc, indices_h, sin_h)
    pos_enc = tf.tensor_scatter_nd_update(pos_enc, indices_h_plus_1, cos_h)

    return pos_enc[tf.newaxis, :]


class ScaledDotProductAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ScaledDotProductAttention, self).__init__(**kwargs)

    def call(self, query, key, value):
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        depth = tf.cast(tf.shape(key)[-1], tf.float32)
        logits = matmul_qk / tf.math.sqrt(depth)
        attention_weights = tf.nn.softmax(logits, axis=-1)
        output = tf.matmul(attention_weights, value)
        return output  # (batch_size, num_attention_heads, sequence_length, depth)


class MultiHeadAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_attention_heads, use_bias=True, **kwargs):
        super(MultiHeadAttentionLayer, self).__init__(**kwargs)
        self.num_attention_heads = num_attention_heads
        self.embed_dim = embed_dim
        self.use_bias = use_bias
        # Ensure the embed_dim is divisible by num_attention_heads
        assert (
            embed_dim % num_attention_heads == 0
        ), f"embed_dim = {embed_dim} should be divisible by num_attention_heads = {num_attention_heads}"

        self.depth = embed_dim // self.num_attention_heads

    def build(self, input_shape):
        self.query_dense = tf.keras.layers.Dense(
            self.embed_dim, kernel_initializer="glorot_uniform", use_bias=self.use_bias
        )
        self.key_dense = tf.keras.layers.Dense(
            self.embed_dim, kernel_initializer="glorot_uniform", use_bias=self.use_bias
        )
        self.value_dense = tf.keras.layers.Dense(
            self.embed_dim, kernel_initializer="glorot_uniform", use_bias=self.use_bias
        )
        self.final_dense = tf.keras.layers.Dense(
            self.embed_dim, kernel_initializer="glorot_uniform", use_bias=self.use_bias
        )
        super(MultiHeadAttentionLayer, self).build(input_shape)

    def split_heads(
        self, x, batch_size
    ):  # input_shape = (batch_size, seq_length, embed_dim)
        x = tf.reshape(x, (batch_size, -1, self.num_attention_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
        # Tensor shape after reshaping and transposing: (batch_size, num_attention_heads, seq_length, depth)

    def call(self, q, k, v):
        batch_size = tf.shape(q)[0]

        # Linear projections
        query = self.query_dense(q)  # (batch_size, seq_length, embed_dim)
        key = self.key_dense(k)
        value = self.value_dense(v)

        # Split heads
        query = self.split_heads(
            query, batch_size
        )  # (batch_size, num_attention_heads, seq_length, depth)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # Scaled dot-product attention
        attention = ScaledDotProductAttention()(
            query, key, value
        )  # (batch_size, num_attention_heads, sequence_length, depth)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, sequence_length, num_attention_heads, depth)

        # Concatenation of heads
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, sequence_length, embed_dim)

        # Final linear projection
        output = self.final_dense(
            concat_attention
        )  # (batch_size, sequence_length, embed_dim)
        return output


class CrossAttentionEncoderLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        dim_embed,
        num_attention_heads,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        layer_norm_eps=1e-5,
        use_layer_scale=False,
        layer_scale_values=1e-4,
        pre_norm=False,
        post_norm=False,
        batch_first=False,
    ):
        """
        Initializes the CrossAttentionEncoderLayer.

        This class constructs a single layer of a Transformer encoder that incorporates
        cross-attention, which allows the layer to attend to a different input sequence
        (key) along with the original input sequence (query). The layer also includes
        a feed-forward neural network and supports various normalization and scaling strategies.

        Args:
            dim_embed (int): Dimension of the embeddings.
            num_attention_heads (int): Number of attention heads.
            dim_feedforward (int): Dimension of the feedforward network. Defaults to 2048.
            dropout (float): Dropout rate. Defaults to 0.1.
            activation (str): Activation function. Should be either "relu" or "gelu". Defaults to "relu".
            norm_epsilon (float): Epsilon value for layer normalization. Defaults to 1e-5.
            use_layer_scale (bool): Whether to use layer scaling. Defaults to False.
            layer_scale_values (float): Initial values for layer scaling. Defaults to 1e-4.
            pre_norm (bool): Whether to apply layer normalization before the sub-layers. Defaults to False.
            post_norm (bool): Whether to apply layer normalization after the sub-layers. Defaults to False.
            batch_first (bool): Whether the input is batch-first format. Defaults to False.

        Raises:
            ValueError: If activation is neither "relu" nor "gelu".
        """
        super(CrossAttentionEncoderLayer, self).__init__()
        self.pre_norm = pre_norm
        self.post_norm = post_norm
        self.batch_first = batch_first

        self.cross_attn = MultiHeadAttentionLayer(
            num_attention_heads=num_attention_heads, embed_dim=dim_embed
        )

        self.feed_forward_layer1 = Dense(dim_feedforward)
        self.dropout = Dropout(dropout)
        self.feed_forward_layer2 = Dense(dim_embed)

        self.norm1 = LayerNormalization(epsilon=layer_norm_eps)
        self.norm2 = LayerNormalization(epsilon=layer_norm_eps)
        self.norm3 = LayerNormalization(epsilon=layer_norm_eps)
        self.post_norm = None
        if self.pre_norm & post_norm:
            self.post_norm = GroupNormalization(
                int(post_norm), axis=-1, epsilon=0.00001
            )

        self.gamma_1 = (
            LayerScale(dim_embed, layer_scale_values, True)
            if use_layer_scale
            else tf.identity
        )
        self.gamma_2 = (
            LayerScale(dim_embed, layer_scale_values, True)
            if use_layer_scale
            else tf.identity
        )

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        if activation == "relu":
            self.activation = tf.nn.relu
        elif activation == "gelu":
            self.activation = tf.nn.gelu
        else:
            raise ValueError(
                "activation should be relu/gelu, not {}".format(activation)
            )

    def call(self, query, key):
        """
        Args:
            q: tensor of shape (T, B, C)
            k: tensor of shape (S, B, C)
            mask: tensor of shape (T, S)
        """
        if self.pre_norm:
            normed_query = self.norm1(query)
            normed_key = self.norm2(key)
            x = query + self.gamma_1(
                self.cross_attention_block(normed_query, normed_key)
            )
            x = x + self.gamma_2(self.feed_forward_block(self.norm3(x)))
            if self.post_norm:
                x = self.post_norm(x)
        else:
            x = self.norm1(query + self.gamma_1(self.cross_attention_block(query, key)))
            x = self.norm2(x + self.gamma_2(self.feed_forward_block(x)))

        return x

    def cross_attention_block(self, query, key):
        x = self.cross_attn(query, key, key)
        return self.dropout1(x)

    def feed_forward_block(self, x):
        x = self.feed_forward_layer2(
            self.dropout(self.activation(self.feed_forward_layer1(x)))
        )
        return self.dropout2(x)


class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        dim_embed,
        num_attention_heads,
        dim_feedforward=2048,
        dropout=0.0,
        activation=tf.nn.relu,
        pre_norm=False,
        post_norm=False,
        layer_norm_eps=1e-5,
        layer_scale_channel_last=True,
        layer_scale_values=1e-4,
        batch_first=False,
        use_layer_scale=True,
    ):
        """
        Initializes the TransformerEncoderLayer.

        This class constructs a single layer of a Transformer encoder, which includes
        multi-head self-attention and a feed-forward neural network. It also supports
        various normalization and scaling strategies.

        Args:
            dim_embed (int): Dimension of the embeddings.
            num_attention_heads (int): Number of attention heads.
            dim_feedforward (int): Dimension of the feedforward network. Defaults to 2048.
            dropout (float): Dropout rate. Defaults to 0.0.
            activation (callable): Activation function. Defaults to tf.nn.relu.
            pre_norm (bool): Whether to apply layer normalization before the sub-layers. Defaults to False.
            post_norm (bool): Whether to apply layer normalization after the sub-layers. Defaults to False.
            layer_norm_eps (float): Epsilon value for layer normalization. Defaults to 1e-5.
            use_layer_scale (bool): Whether to use layer scaling. Defaults to True.
            layer_scale_values (float): Initial values for layer scaling. Defaults to 1e-4.
            layer_scale_channel_last (bool): Whether the layer scaling is applied in channel_last mode. Defaults to True.
            batch_first (bool): Whether the input is batch-first format. Defaults to False.

        Raises:
            AssertionError: If embedding_dim is not divisible by num_attention_heads.
        """
        super(TransformerEncoderLayer, self).__init__()
        self.dim_embed = dim_embed
        self.nhead = num_attention_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.pre_norm = pre_norm
        self.post_norm = post_norm
        self.layer_norm_eps = layer_norm_eps
        self.layer_scale_channel_last = layer_scale_channel_last
        self.layer_scale_values = layer_scale_values
        self.batch_first = batch_first

        self.self_attn = MultiHeadAttentionLayer(
            num_attention_heads=num_attention_heads, embed_dim=dim_embed
        )
        self.ffn = tf.keras.Sequential(
            [
                Dense(dim_feedforward, activation=activation),
                Dropout(dropout),
                Dense(dim_embed),
                Dropout(dropout),
            ]
        )
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.norm1 = LayerNormalization(epsilon=layer_norm_eps)
        self.norm2 = LayerNormalization(epsilon=layer_norm_eps)

        if self.pre_norm and self.post_norm:
            self.post_norm = GroupNormalization(
                groups=int(post_norm), axis=-1, epsilon=0.00001
            )

        self.gamma_1 = (
            LayerScale(dim_embed, layer_scale_values, layer_scale_channel_last)
            if use_layer_scale
            else tf.identity
        )
        self.gamma_2 = (
            LayerScale(dim_embed, layer_scale_values, layer_scale_channel_last)
            if use_layer_scale
            else tf.identity
        )

    def self_attention_block(self, x):
        attn_output = self.self_attn(x, x, x)
        return self.dropout1(attn_output)

    def _ff_block(self, x):
        x = self.ffn(x)
        return self.dropout2(x)

    def call(self, src, mask=None, training=False):
        if self.batch_first:
            x = src
        else:
            x = tf.transpose(src, perm=[1, 0, 2])  # Change (T, B, C) to (B, T, C)

        if self.pre_norm:
            x = x + self.gamma_1(self.self_attention_block(self.norm1(x)))
            x = x + self.gamma_2(self._ff_block(self.norm2(x)))

            if self.post_norm:
                x = self.post_norm(x)

        else:
            self_attention_block_out = x + self.gamma_1(self.self_attention_block(x))
            x = self.norm1(self_attention_block_out)
            feed_forward_block_out = x + self.gamma_2(self._ff_block(x))
            x = self.norm2(feed_forward_block_out)

        if not self.batch_first:
            x = tf.transpose(x, perm=[1, 0, 2])  # Change (B, T, C) back to (T, B, C)

        return x


def get_pos_embedding(time_dim_wave, channel):
    # shift = random.randrange(self.sin_random_shift +1) #currently we not utilizing this
    shift = 0
    pos_emb = create_sin_embedding(time_dim_wave, channel, shift)
    return pos_emb


# @keras.saving.register_keras_serializable()
class CrossDomainTransformerEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
        dim_embed,
        hidden_scale=4.0,
        num_attention_heads=8,
        num_layers=6,
        cross_attention_first=False,
        dropout=0.0,
        normalize_input=True,
        pre_norm=False,
        post_norm=False,
        use_layer_scale=False,
        activation="gelu",
        max_period: float = 10000.0,
        weight_pos_embed: float = 1.0,
        sin_random_shift: int = 0,
    ):
        """
        Initializes the CrossDomainTransformerEncoder.

        This class performs both classic Transformer encoding and cross-attention encoding
        between frequency domain and time domain inputs. It allows flexible configuration
        for the number of layers, attention heads, activation functions, normalization,
        and other aspects to suit various applications and domain-specific needs.

        Arguments:
        dim_embed -- Dimension of the embeddings.
        hidden_dim_scale -- Scale factor for the hidden dimension.
        num_attention_heads -- Number of attention heads.
        num_layers -- Number of transformer layers.
        cross_attention_first -- Whether cross-attention layers come first.
        dropout -- Dropout rate.
        normalize_input -- Whether to normalize the input.
        pre_norm -- Whether to apply layer normalization before the transformer sub-layers.
        post_norm -- Whether to apply layer normalization after the transformer sub-layers.
        use_layer_scale -- Whether to use layer scaling.
        activation -- Activation function to use.
        max_period -- Maximum period of position encoding.
        weight_pos_embed -- Weighting factor for positional embeddings.
        sinusoid_random_shift -- Random shift for sinusoidal embeddings.
        """
        super(CrossDomainTransformerEncoder, self).__init__()

        assert dim_embed % num_attention_heads == 0

        hidden_dim = int(dim_embed * hidden_scale)

        self.num_layers = num_layers
        self.classic_parity = 1 if cross_attention_first else 0
        self.max_period = max_period
        self.weight_pos_embed = weight_pos_embed
        self.sin_random_shift = sin_random_shift

        self.normalize_input = (
            LayerNormalization(epsilon=1e-5) if normalize_input else tf.identity
        )
        self.normalize_input_time_domain = (
            LayerNormalization(epsilon=1e-5) if normalize_input else tf.identity
        )

        self.layers = []
        self.layers_t = []

        kwargs_common = {
            "dim_embed": dim_embed,
            "num_attention_heads": num_attention_heads,
            "dim_feedforward": hidden_dim,
            "dropout": dropout,
            "activation": activation,
            "pre_norm": pre_norm,
            "post_norm": post_norm,
            "use_layer_scale": use_layer_scale,
            "batch_first": True,
        }

        for idx in range(num_layers):
            if idx % 2 == self.classic_parity:
                self.layers.append(TransformerEncoderLayer(**kwargs_common))
                self.layers_t.append(TransformerEncoderLayer(**kwargs_common))
            else:
                self.layers.append(CrossAttentionEncoderLayer(**kwargs_common))
                self.layers_t.append(CrossAttentionEncoderLayer(**kwargs_common))

    def call(self, frequency_domain_tensor, time_domain_tensor, training=False):
        batch, channel, frequency, frames = (
            tf.shape(frequency_domain_tensor)[0],
            tf.shape(frequency_domain_tensor)[1],
            tf.shape(frequency_domain_tensor)[2],
            tf.shape(frequency_domain_tensor)[3],
        )
        pos_emb_2d = create_2d_sin_embedding(
            channel, frequency, frames, self.max_period
        )
        pos_emb_2d = einops.rearrange(pos_emb_2d, "b c fr t1 -> b (t1 fr) c")
        frequency_domain_tensor = einops.rearrange(
            frequency_domain_tensor, "b c fr t1 -> b (t1 fr) c"
        )
        frequency_domain_tensor = self.normalize_input(frequency_domain_tensor)
        frequency_domain_tensor = (
            frequency_domain_tensor + self.weight_pos_embed * pos_emb_2d
        )

        batch, channel, time_dim_wave = (
            tf.shape(time_domain_tensor)[0],
            tf.shape(time_domain_tensor)[1],
            tf.shape(time_domain_tensor)[2],
        )
        time_domain_tensor = einops.rearrange(time_domain_tensor, "b c t2 -> b t2 c")
        pos_emb = get_pos_embedding(time_dim_wave, channel)
        pos_emb = einops.rearrange(pos_emb, "t2 b c -> b t2 c")
        time_domain_tensor = self.normalize_input_time_domain(time_domain_tensor)
        time_domain_tensor = time_domain_tensor + self.weight_pos_embed * pos_emb

        for idx in range(self.num_layers):
            if idx % 2 == self.classic_parity:
                frequency_domain_tensor = self.layers[idx](
                    frequency_domain_tensor, training=training
                )
                time_domain_tensor = self.layers_t[idx](
                    time_domain_tensor, training=training
                )
            else:
                old_frequency_domain_tensor = frequency_domain_tensor
                frequency_domain_tensor = self.layers[idx](
                    frequency_domain_tensor, time_domain_tensor, training=training
                )
                time_domain_tensor = self.layers_t[idx](
                    time_domain_tensor, old_frequency_domain_tensor, training=training
                )
        frequency_domain_tensor = einops.rearrange(
            frequency_domain_tensor, "b (t1 fr) c -> b c fr t1", t1=frames
        )
        time_domain_tensor = einops.rearrange(time_domain_tensor, "b t2 c -> b c t2")
        return frequency_domain_tensor, time_domain_tensor


def test_multi_head_attention():
    # Load weights
    weights = np.load("initial_weights.npz")

    # TensorFlow uses a different ordering, so you might need to transpose matrices
    weights_q = np.transpose(weights["weights_q"])
    weights_k = np.transpose(weights["weights_k"])
    weights_v = np.transpose(weights["weights_v"])
    weights_proj = np.transpose(weights["weights_proj"])

    # Load input data
    input_data = np.load("initial_input.npy")
    input_tensor = tf.constant(input_data)

    embed_dim = 2
    num_attention_heads = 2
    seq_len = 5

    multi_head_attn = MultiHeadAttentionLayer(
        embed_dim, num_attention_heads, use_bias=False
    )

    multi_head_attn.build(
        input_shape=(None, seq_len, embed_dim)
    )  # batch_size, seq_len, embed_dim
    output = multi_head_attn(input_tensor, input_tensor, input_tensor)

    multi_head_attn.query_dense.set_weights([weights_q.T])
    multi_head_attn.key_dense.set_weights([weights_k.T])
    multi_head_attn.value_dense.set_weights([weights_v.T])
    multi_head_attn.final_dense.set_weights([weights_proj.T])

    output = multi_head_attn(input_tensor, input_tensor, input_tensor)
    print("Output: ", output)  # Should be (batch_size, seq_len, embed_dim)


def test_positional_encoding():
    length, dim, shift = 2, 4, 0
    tf_emb = create_sin_embedding(length, dim, shift)
    print(tf_emb)
    dim_emb = 8
    height = 2
    width = 3

    out_tf = create_2d_sin_embedding(dim_emb, height, width)

    print("Output ", out_tf)


def test_transformer_encoder_layer():
    input_tensor = np.random.rand(2, 5, 4).astype(np.float32)

    # tf_model = TransformerEncoderLayer(dim_embed=4, num_attention_heads=2, batch_first=True)
    tf_model = CrossAttentionEncoderLayer(dim_embed=4, num_attention_heads=2)
    # Convert input data to tf tensor for TensorFlow model
    tf_input = tf.convert_to_tensor(input_tensor)

    # Run the TensorFlow model
    tf_output = tf_model(tf_input, tf_input).numpy()
    print("Output ", tf_output, tf_output.shape)


def test_layer_scale():
    channels = 4
    init_value = 0.5
    channel_last = True
    layer_tf = LayerScale(channels, init_value, channel_last)

    # Create a random tensor input
    input_data_np = np.random.rand(2, 3, 4).astype(
        np.float32
    )  # Example shape (B, T, C)
    # Convert the NumPy array to the respective framework's tensor
    input_data_tf = tf.convert_to_tensor(input_data_np)
    output_tf = layer_tf(input_data_tf).numpy()
    print("Output ", output_tf)


def test_cross_domain_transformer_encoder_layer():
    encoder = CrossDomainTransformerEncoder(
        dim_embed=64,  # Small embedding size for testing
        num_attention_heads=4,  # Number of attention heads
        num_layers=2,  # Number of transformer layers
        cross_attention_first=True,  # Whether to apply cross attention first
        dropout=0.1,  # Dropout rate
        weight_pos_embed=1.0,  # Position embedding weight
    )

    # Create simple inputs
    batch_size = 2
    C = 64  # Embedding dimension
    Fr = 10  # Frequency dimension for input x
    time_dim_fr = 20  # Time dimension for input x
    time_dim_wave = 15  # Time dimension for input time_domain_tensor

    frequency_domain_tensor = tf.random.uniform(
        (batch_size, C, Fr, time_dim_fr), dtype=tf.float32
    )
    time_domain_tensor = tf.random.uniform(
        (batch_size, C, time_dim_wave), dtype=tf.float32
    )

    # Forward pass
    output_frequency_domain_tensor, output_time_domain_tensor = encoder(
        frequency_domain_tensor, time_domain_tensor
    )
    print("Output frequency_domain_tensor shape:", output_frequency_domain_tensor.shape)
    print("Output time_domain_tensor shape:", output_time_domain_tensor.shape)


if __name__ == "__main__":
    # test_positional_encoding()
    # test_layer_scale()
    # test_multi_head_attention()
    test_transformer_encoder_layer()
    # test_cross_domain_transformer_encoder_layer()
