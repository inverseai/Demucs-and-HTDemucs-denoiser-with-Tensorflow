import tensorflow as tf
import numpy as np
from demucs import Demucs

model_path = "/home/ml-dev/Noise_reducer/out/checkpoint_epoch-007_loss-0.002056_val_loss-0.002355.h5"
total_number_of_sample = 160000
model = Demucs(input_shape=(total_number_of_sample, 1))
model.load_weights(model_path)

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)



# Testing tflite model
interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
interpreter.allocate_tensors()

# Print input details
input_details = interpreter.get_input_details()
print("Input details:", input_details)

# Print output details
output_details = interpreter.get_output_details()
print("Output details:", output_details)

# Prepare sample input data
input_shape = input_details[0]['shape']
input_data = np.random.rand(*input_shape).astype(np.float32)

print("Input ",input_data.shape)

# Set the input tensor to the sample data
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Get the output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])

# Print the output
print("Output:", output_data)