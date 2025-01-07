import tensorflow as tf
from htdemucs import HTDemucs
import load
from names_map import mapping
import librosa


def test_htdemucs():
    batch_size = 1
    sample = 160000
    channel = 1
    input_shape = [batch_size, sample, channel]

    MODE = 0
    mode = ["build", "saved", "weight"][MODE]

    if mode == "build":
        model = HTDemucs(**load.config)
        model.build(input_shape)
        weight_dict = load.load_weights_dict_from_h5(
            "/mnt/4tb/Saiham/htd/best_fixed.h5"
        )
        load.load_and_validate_weights(model, weight_dict, mapping)

    elif mode == "saved":
        model = tf.saved_model.load("/mnt/4tb/Saiham/htd/saved_model")

    elif mode == "weight":
        model = HTDemucs(**load.config)
        model.build(input_shape)
        model.load_weights("/mnt/4tb/Saiham/htd/checkpoints.weights.h5")

    # print(model.summary())

    # for layer in model.layers:
    #     weights = layer.weights
    #     if weights:
    #         print("\033[36m" + f"Layer: {layer.name}" + "\033[0m")
    #         for obj in weights:
    #             print(f"{obj.path:<90}")

    input_dir = "/home/saiham/Music/music/"
    output_dir = "/home/saiham/Music/music/"
    import soundfile as sf
    import os

    def process_file(file_path, model):
        signal = librosa.load(file_path, sr=16000)[0]
        input_tensor = tf.reshape(signal, [-1, sample, channel])
        out = model(input_tensor)

        output_path = os.path.join(
            output_dir, os.path.basename(file_path).replace(".wav", ".wav")
        )
        out = tf.reshape(out, [-1])
        sf.write(output_path, out.numpy().squeeze(), 16000)

    # Traverse the directory and process each audio file
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".wav"):
            file_path = os.path.join(input_dir, file_name)
            process_file(file_path, model=model.serve if mode == "saved" else model)

    if mode == "build":
        model.export("/mnt/4tb/Saiham/htd/saved_model")
        model.save_weights("/mnt/4tb/Saiham/htd/checkpoints.weights.h5")


if __name__ == "__main__":
    test_htdemucs()
