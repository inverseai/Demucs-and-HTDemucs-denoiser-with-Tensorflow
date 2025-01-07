import math
import os
import soundfile as sf
import numpy as np
import tensorflow as tf

tflite_dir = "/mnt/4tb/Saiham/htd/model_float16.tflite"
saved_model_dir = "/mnt/4tb/Saiham/htd/saved_model"


class TestModel:
    def __init__(self, tflite_dir=None, tfmodel_dir=None) -> None:
        if tflite_dir:
            self.interpreter = tf.lite.Interpreter(model_path=tflite_dir)

            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        if tfmodel_dir:
            self.model = tf.saved_model.load("/mnt/4tb/Saiham/htd/saved_model")

    def run_lite_model(self, input_audio):
        input_shape = self.input_details[0]["shape"]
        input_data = np.array(input_audio, dtype=np.float32)
        # Resizing the input tensor to match the current batch's shape
        self.interpreter.resize_tensor_input(
            self.input_details[0]["index"], input_data.shape
        )
        self.interpreter.allocate_tensors()
        # input_data = np.reshape(input_data, input_shape)
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)

        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.output_details[0]["index"])
        return output_data

    def run_tf_model(self, input_audio):
        input_data = tf.convert_to_tensor(input_audio, dtype=tf.float32)
        output_data = self.model.serve(input_data)
        return output_data.numpy()


tester = TestModel(tflite_dir=tflite_dir, tfmodel_dir=None)

input_dir = "/home/saiham/Music/music/"
output_dir = "/home/saiham/Music/music/"
os.makedirs(output_dir, exist_ok=True)

batch_size = 3  # Define suitable batch size
segment_length = 160000  # Length of each segment
total_files = len(os.listdir(input_dir))

import time

begin_time = time.time()
for i, filename in enumerate(os.listdir(input_dir)):

    input_audio, _ = sf.read(os.path.join(input_dir, filename))
    # Pad with 0s so that length is multiple of segment_length
    padding = math.ceil(len(input_audio) / segment_length) * segment_length - len(
        input_audio
    )
    input_audio = np.pad(input_audio, (0, padding), "constant")
    segments = math.ceil(len(input_audio) / segment_length)
    input_segments = np.array_split(input_audio, segments)

    all_outputs = []

    # Process in batches
    for j in range(0, segments, batch_size):
        batch_segments = input_segments[j : j + batch_size]
        batch_input = np.array(batch_segments, dtype=np.float32)
        batch_input = np.expand_dims(
            batch_input, axis=-1
        )  # Shape to (batch_size, segment_length, 1)
        batch_output = tester.run_lite_model(batch_input)
        all_outputs.append(batch_output)
        print(len(all_outputs))

    output_audio = np.concatenate(all_outputs, axis=0)
    b, s, c = output_audio.shape
    output_audio = np.reshape(output_audio, [b * s * c])
    if padding > 0:
        output_audio = output_audio[:-padding]

    sf.write(os.path.join(output_dir, filename), output_audio, 16000, format="wav")

    print(f"{i+1}/{total_files} done")

print("Infer time ", time.time() - begin_time)
