import json
import os
import tensorflow as tf
from htdemucs import HTDemucs
import load

os.environ["CUDA_VISIBLE_DEVICES"] = ""

def generate_dataset(x_dir, y_dir):
    dataset = lambda dir: tf.keras.utils.audio_dataset_from_directory(
        directory=dir,
        batch_size=2,
        labels=None,
        seed=0,
        validation_split=0.0,
        subset=None,
        output_sequence_length=160000,
    )
    train_x = dataset(x_dir)
    train_y= dataset(y_dir)

    reshape = lambda x: tf.reshape(x, [-1, 160000, 1])

    train_x = train_x.map(reshape)
    #val_x = val_x.map(reshape)
    train_y = train_y.map(reshape)
    #val_y = val_y.map(reshape)

    train_ds = tf.data.Dataset.zip((train_x, train_y))
    #val_ds = tf.data.Dataset.zip((val_x, val_y))

    return train_ds


# Load audio files
clean_folder = "/home/inverseai/Data/sample_train_data/clean/"
noisy_folder = "/home/inverseai/Data/sample_train_data/noisy/"

train_dataset = generate_dataset(noisy_folder, clean_folder)

# save dataset as tfrecord
# train_dataset.save("train.tfrecord")
# val_dataset.save("val.tfrecord")

# Load the pre-trained model
input_shape = (1, 160000, 1)
model = HTDemucs(**load.config)
model.build(input_shape=input_shape)
#model.load_weights("/mnt/Data_storage_SSD/saves/checkpoints.weights.h5")

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=3e-4,
    ),
    loss="mae",
)

# Define the checkpoint callback
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="/home/inverseai/Data/sample_train_data/checkpoints/checkpoint_epoch_{epoch:02d}.keras",
    save_best_only=False,
    #save_freq="epoch",
    save_freq=100  # Save checkpoints every 4 epochs
)

# print("Training...")
# Fit the model
result = model.fit(
    x=train_dataset,
   #validation_data=val_dataset,
    epochs=500,
    callbacks=[checkpoint_callback],
)

print("Done training!")
with open("history.json", "w") as f:
    json.dump(result.history, f)