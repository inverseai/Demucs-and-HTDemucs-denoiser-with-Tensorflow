import os
import re
import yaml
import tensorflow as tf

from denoiser.demucs import Demucs
from denoiser.stft_loss import custom_loss
from denoiser.htdemucs import HTDemucs
from data_load_and_preprocess import get_dataset_train_raw, get_dataset_val_raw


def load_config(config_file):
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def get_model(model_name, config):
    print(config["sample_rate"])
    if model_name == "demucs":
        model = Demucs(input_shape=(160000, 1))
    elif model_name == "htdemucs":
        model = HTDemucs(
            **config["models"]["htdemucs"],
            sample_rate=config["sample_rate"],
            length=config["total_seconds_of_audio"],
        )
        model.build(
            input_shape=(
                config["batch_size"],
                config["sample_rate"] * config["total_seconds_of_audio"],
                config["channel"],
            )
        )
    return model


def _parse_batch(record_batch, sample_rate, duration, split):
    n_samples = sample_rate * duration
    feature_description = {
        "noisy": tf.io.FixedLenFeature([n_samples], tf.float32),
        "clean": tf.io.FixedLenFeature([n_samples], tf.float32),
    }
    example = tf.io.parse_example(record_batch, feature_description)
    noisy, clean = tf.expand_dims(example["noisy"], axis=-1), tf.expand_dims(
        example["clean"], axis=-1
    )

    # if split == 'train':
    #     noisy, clean = augment(noisy, clean)
    return noisy, clean

def get_dataset_from_tfrecords(config, tfrecords_dir='/content/sample_data/tfrecords', split='train',
                               batch_size=64, sample_rate=16000, duration=4, AUTOTUNE=tf.data.experimental.AUTOTUNE):
    if split not in ('train', 'test', 'validate'):
        raise ValueError("split must be either 'train', 'test' or 'validate'")
    pattern = os.path.join(tfrecords_dir, "{}*.tfrecord".format(split))
    files_ds = tf.data.Dataset.list_files(pattern)
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    files_ds = files_ds.with_options(ignore_order)
    ds = tf.data.TFRecordDataset(files_ds, compression_type='ZLIB', num_parallel_reads=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.map(lambda x: _parse_batch(x, sample_rate, duration, split))
    if config["steps_per_epoch"] != -1:
        ds = ds.repeat()
    return ds.prefetch(buffer_size=AUTOTUNE)


def get_compiled_model(config, BATCH_SIZE, compiled=True):
    model = get_model(config["model_name"], config)
    print(model.summary())

    if compiled:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=config["learning_rate"],
                beta_1=config["beta_1"],
                beta_2=config["beta_2"],
            ),
            loss="mae",
            # loss=custom_loss(BATCH_SIZE)  # for demucs
        )
    return model


def make_or_restore_model(config, BATCH_SIZE, compiled=True):
    checkpoints = [
        os.path.join(config["checkpoints_dir"], name)
        for name in os.listdir(config["checkpoints_dir"])
    ]

    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        latest_epoch = int(re.search(r"epoch-(\d+)", latest_checkpoint).group(1))
        model = tf.keras.models.load_model(
            latest_checkpoint,
            custom_objects={"custom_loss": custom_loss, "BATCH_SIZE": BATCH_SIZE},
        )
    else:
        model = get_compiled_model(config, BATCH_SIZE, compiled=compiled)
        latest_epoch = 0

    return model, latest_epoch


def get_dataset_train_tfrecords(config, BATCH_SIZE):
    return get_dataset_from_tfrecords(
        config,
        tfrecords_dir=config["tfrecords_dir"],
        batch_size=BATCH_SIZE,
        sample_rate=config["sample_rate"],
        duration=config["total_seconds_of_audio"],
        AUTOTUNE=(tf.data.AUTOTUNE if config["autotune"] == -1 else config["autotune"]),
    )


def get_dataset_val_tfrecords(config, BATCH_SIZE):
    return get_dataset_from_tfrecords(
        config,
        tfrecords_dir=config["tfrecords_dir"],
        batch_size=BATCH_SIZE,
        split="validate",
        sample_rate=config["sample_rate"],
        duration=config["total_seconds_of_audio"],
        AUTOTUNE=(
            tf.data.experimental.AUTOTUNE
            if config["autotune"] == -1
            else config["autotune"]
        ),
    )


def run_training(config, steps="ckpt"):
    strategy = tf.distribute.MirroredStrategy()
    BATCH_SIZE_PER_REPLICA = config["batch_size"]
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    config["autotune"] = BATCH_SIZE * 2
    print("Batch size:", BATCH_SIZE)
    print(f"Number of GPUs we have: {strategy.num_replicas_in_sync}")

    validation_dataset = None
    if config["val_freq"] != 0:
        if config["dataset_format"] == "raw":
            validation_dataset = get_dataset_val_raw(config, BATCH_SIZE)
        else:
            validation_dataset = get_dataset_val_tfrecords(config, BATCH_SIZE)

    import time

    begin_time = time.time()
    with strategy.scope():
        model, latest_epoch = make_or_restore_model(config, BATCH_SIZE_PER_REPLICA)
        if config["dataset_format"] == "raw":
            train_dataset = get_dataset_train_raw(config, BATCH_SIZE)
        else:
            train_dataset = get_dataset_train_tfrecords(config, BATCH_SIZE)

        for batch in train_dataset.take(1):
            print([x.shape for x in batch])

        train_dataset = strategy.experimental_distribute_dataset(train_dataset)

        if config["val_freq"] == 1:
            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(
                        config["checkpoints_dir"],
                        f"{steps}_epoch-{{epoch:03d}}_loss-{{loss:.6f}}_val_loss-{{val_loss:.6f}}.keras",
                    ),
                    save_best_only=False,
                    save_weights_only=False,
                )
            ]
        else:
            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(
                        config["checkpoints_dir"],
                        f"{steps}_epoch-{{epoch:03d}}_loss-{{loss:.6f}}.keras",
                    ),
                    save_best_only=False,
                    save_weights_only=False,
                )
            ]

        history = model.fit(
            train_dataset,
            epochs=config["epochs"],
            initial_epoch=latest_epoch,
            callbacks=callbacks,
            validation_data=validation_dataset if config["val_freq"] != 0 else None,
            steps_per_epoch=(
                None if config["steps_per_epoch"] == -1 else config["steps_per_epoch"]
            ),
            validation_steps=(
                config["val_steps_per_epoch"] if config["val_freq"] != 0 else None
            ),
            validation_freq=config["val_freq"] if config["val_freq"] != 0 else None,
            verbose=1,
        )
        # model.summary()
        # print(model.loss)
        return
        # print(history.history)
    print("Total time ", time.time() - begin_time)
    print("done\n\n")
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default="config.yaml",
        help="Path to the config file",
    )
    args = parser.parse_args()

    config = load_config(args.config_file)

    if not os.path.exists(config["checkpoints_dir"]):
        os.makedirs(config["checkpoints_dir"])

    model = run_training(config, f"checkpoint")
