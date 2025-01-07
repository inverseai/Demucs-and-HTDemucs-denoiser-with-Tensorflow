import h5py
import tensorflow as tf


def save_weights_to_h5(pytorch_model, h5_filepath):
    # Create an HDF5 file
    with h5py.File(h5_filepath, "w") as h5f:
        for name, param in pytorch_model.named_parameters():
            # Convert PyTorch tensor to NumPy array
            if param.dim() == 3:  # time domain
                t = param.permute(2, 1, 0)
                # print(f"{name:<60}  |   {t.shape}")
                h5f.create_dataset(name, data=t.detach().cpu().numpy())
                # print(param.shape, t.shape)
            elif param.dim() == 4:  # frequency domain
                t = param.permute(2, 3, 1, 0)
                # print(f"{name:<60}  |   {t.shape}")
                h5f.create_dataset(name, data=t.detach().cpu().numpy())
            elif param.shape[0] == 1152:
                embed_dim = 384
                name = ".".join(name.split(".")[:-1])
                if param.dim() == 2:
                    q = name + ".query.weight"
                    k = name + ".key.weight"
                    v = name + ".value.weight"
                    query = param[:embed_dim, :]
                    key = param[embed_dim : 2 * embed_dim, :]
                    value = param[2 * embed_dim :, :]
                else:
                    q = name + ".query.bias"
                    k = name + ".key.bias"
                    v = name + ".value.bias"
                    query = param[:embed_dim]
                    key = param[embed_dim : 2 * embed_dim]
                    value = param[2 * embed_dim :]
                # print(f"{q:<60}  |   {query.shape}")
                # print(f"{k:<60}  |   {key.shape}")
                # print(f"{v:<60}  |   {value.shape}")
                h5f.create_dataset(q, data=query.detach().cpu().numpy())
                h5f.create_dataset(k, data=key.detach().cpu().numpy())
                h5f.create_dataset(v, data=value.detach().cpu().numpy())
                # print(param.shape, t.shape)
            elif param.dim() == 2 and (param.shape[1] == 384 or param.shape[0] == 384):
                t = param.permute(1, 0)
                # print(f"{name:<60}  |   {t.shape}")
                h5f.create_dataset(name, data=t.detach().cpu().numpy())
            else:
                # print(f"{name:<60}  |   {param.shape}")
                param_np = param.detach().cpu().numpy()
                # Save into HDF5 file
                h5f.create_dataset(name, data=param_np)


def load_weights_dict_from_h5(h5_filepath):
    """
    Load weights from an HDF5 file into a dictionary.

    Args:
        h5_filepath (str): The path to the HDF5 file containing the weights.

    Returns:
        dict: A dictionary containing the weights, where the keys are the weight names and the values are the corresponding weight values.
    """
    weights_dict = {}
    with h5py.File(h5_filepath, "r") as h5f:
        for key in h5f.keys():
            print(type(h5f[key][()]))
            weights_dict[key] = h5f[key][()]
    return weights_dict


# ================== BROKEN =================
# def load_weights_from_h5(model, h5_filepath):
#     """
#     Load weights from an HDF5 file into a Keras model.

#     Args:
#         model (tf.keras.Model): The Keras model to load weights into.
#         h5_filepath (str): The path to the HDF5 file containing the weights.

#     Returns:
#         None
#     """
#     with h5py.File(h5_filepath, "r") as h5f:
#         for layer in model.layers:
#             for weight in layer.weights:
#                 weight_name = weight.name.split(":")[
#                     0
#                 ]  # Remove ':0' from the end of the name
#                 if weight_name in h5f:
#                     weight_data = h5f[weight_name][()]
#                     weight.assign(weight_data)
#                 else:
#                     print(f"Warning: {weight_name} not found in the HDF5 file.")


def load_weights_dict_from_h5(file_path):
    """
    Load weights from an HDF5 file and return them as a dictionary.

    Parameters:
    file_path (str): The path to the HDF5 file containing the weights.

    Returns:
    dict: A dictionary with weights.
    """
    weights_dict = {}

    def recursively_load_group(h5_group, target_dict):
        """Recursively load models weights from an h5 group into provided dictionary."""
        for key, item in h5_group.items():
            if isinstance(item, h5py.Dataset):
                target_dict[key] = item[()]
            elif isinstance(item, h5py.Group):
                target_dict[key] = {}
                recursively_load_group(item, target_dict[key])

    with h5py.File(file_path, "r") as h5_file:
        recursively_load_group(h5_file, weights_dict)

    return weights_dict

def load_and_validate_weights(model, weights_dict, mapping):
    for layer in model.layers:
        # print("\033[36m" + f"Layer: {layer.name}" + "\033[0m")
        new_weights = []
        for obj in layer.weights:
            name = obj.path

            mapped_name = mapping[name]
            if mapped_name in weights_dict:
                val = weights_dict[mapped_name]
                # print(name, "==>", mapped_name)
                tensor = tf.convert_to_tensor(val)
                assert (
                    tensor.shape == obj.shape
                ), f"Weights shape mismatch for {name}: expected {obj.shape}, got {tensor.shape}"
                obj.assign(tensor)
                assert (
                    tensor.numpy() == obj.numpy()
                ).all(), f"Weights mismatch for {name}"
            else:
                print(f"Warning: {mapped_name} not found in weights_dict")


config = {
    "channels": 48,
    "channels_time": None,
    "growth": 2,
    # STFT
    "nfft": 4096,
    "wiener_iters": 0,
    "end_iters": 0,
    "wiener_residual": False,
    "cac": True,
    # Main structure
    "depth": 4,
    "rewrite": True,
    # Frequency Branch
    "multi_freqs": [],
    "multi_freqs_depth": 3,
    "freq_emb": 0.2,
    "emb_scale": 10,
    "emb_smooth": True,
    # Convolutions
    "kernel_size": 8,
    "stride": 4,
    "time_stride": 2,
    "context": 1,
    "context_enc": 0,
    # normalization
    "norm_starts": 4,
    "norm_groups": 4,
    # DConv residual branch
    "dconv_mode": 1,
    "dconv_depth": 2,
    "dconv_comp": 8,
    "dconv_init": 1e-3,
    # CrossTransformer
    # ------ Common to all
    # Regular parameters
    "t_layers": 5,
    "t_hidden_scale": 4.0,
    "t_heads": 8,
    "t_dropout": 0.0,
    "t_layer_scale": True,
    "t_gelu": "gelu",
    # ------------- Positional Embedding
    "t_emb": "sin",
    "t_max_positions": 10000,  # for the scaled embedding
    "t_max_period": 10000.0,
    "t_weight_pos_embed": 1.0,
    "t_cape_mean_normalize": True,
    "t_cape_augment": True,
    "t_cape_glob_loc_scale": [5000.0, 1.0, 1.4],
    "t_sin_random_shift": 0,
    # ------------- norm before a transformer encoder
    "t_norm_in": True,
    "t_norm_in_group": False,
    # ------------- norm inside the encoder
    "t_group_norm": False,
    "t_norm_first": True,
    "t_norm_out": True,
    # ------------- optim
    "t_weight_decay": 0.0,
    "t_lr": None,
    # ------------- sparsity
    "t_sparse_self_attn": False,
    "t_sparse_cross_attn": False,
    "t_mask_type": "diag",
    "t_mask_random_seed": 42,
    "t_sparse_attn_window": 400,
    "t_global_window": 100,
    "t_sparsity": 0.95,
    "t_auto_sparsity": False,
    # Cross Encoder First
    "t_cross_first": False,
}

# # Initialize an empty Keras model
# model = tf.keras.Model()

# # Build the model with the desired input shape
# model.build(input_shape=(1, 160000, 1))

# # Ensure to replace with your model architecture
# # Example of a minimal model that matches the input shape for demonstration purposes
# # Replace this with your actual model definition
# model = tf.keras.Sequential(
#     [
#         tf.keras.layers.InputLayer(input_shape=(160000, 1)),
#         # Add your layers here
#     ]
# )

# # Load weights from HDF5 file
# load_weights_from_h5(model, "/mnt/4tb/Saiham/htd/best.h5")

# Example usage to load weights from an HDF5 file without tying them to a specific model
# weights_dict = load_weights_dict_from_h5('/mnt/4tb/Saiham/htd/best.weights.h5')
# print(weights_dict)
# from names_map import mapping
# Print the keys and shapes of the loaded weights
# print(all(value in mapping.values() for value in weights_dict.keys()))


# for layer in model.layers:
#     weights = layer.weights
#     if weights:
#         print("\033[36m" + f"Layer: {layer.name}" + "\033[0m")
#         # print("Total number of parameters:", sum([p.numel() for p in layer.weights]))
#         for obj in weights:
#             print(f"{obj.path:<88}  |   {obj.shape}")
