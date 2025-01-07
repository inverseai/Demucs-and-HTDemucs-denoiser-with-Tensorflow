import numpy as np
import tensorflow as tf

saved_model_dir = "/mnt/4tb/Saiham/htd/saved_model"
tflite_dir = "/mnt/4tb/Saiham/htd/model_float16.tflite"

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir=saved_model_dir)

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS,  # enable TensorFlow ops.
]

# Require this two lines for doing float16 quantization, otherwise require to skip this two lines
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

# Save the model.
with open(tflite_dir, "wb") as f:
    f.write(tflite_model)

# Testing tflite model
interpreter = tf.lite.Interpreter(model_path=tflite_dir)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
print("Input details:", input_details)

output_details = interpreter.get_output_details()
print("Output details:", output_details)

input_shape = input_details[0]["shape"]
input_data = np.random.rand(*input_shape).astype(np.float32)
interpreter.set_tensor(input_details[0]["index"], input_data)

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]["index"])

print("Output:", output_data.shape)
