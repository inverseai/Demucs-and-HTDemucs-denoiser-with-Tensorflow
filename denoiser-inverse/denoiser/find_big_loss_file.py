import tensorflow as tf
from demucs import Demucs
import argparse
import os
from stft_loss import custom_loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, dest='model_path',
                        default='', help='path of the weighted model'
                                         '(default: %(default)s)')
    parser.add_argument('--tfrecord_path', type=str, dest='tfrecord_path',
                        default='', help='path of the noisy sample'
                                         '(default: %(default)s)')
    return parser.parse_args()


def _parse_batch(record_batch, sample_rate, duration, split):
    n_samples = sample_rate * duration

    # Create a description of the features
    feature_description = {
        'noisy': tf.io.FixedLenFeature([n_samples], tf.float32),
        'clean': tf.io.FixedLenFeature([n_samples], tf.float32),
    }
    # Parse the input `tf.Example` proto using the dictionary above
    example = tf.io.parse_example(record_batch, feature_description)
    # print("Len ",example)
    noisy, clean = tf.expand_dims(example['noisy'], axis=-1), tf.expand_dims(example['clean'], axis=-1)
    return noisy, clean,


def mean_absolute_error_loss(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    loss = tf.keras.metrics.mean_absolute_error(y_true, y_pred)  # (B,) shape
    return loss


def pred(model, noisy):
    print(type(noisy))
    self_sample_rate = 16_000
    total_seconds_of_audio = 10
    total_number_of_sample = self_sample_rate * total_seconds_of_audio
    number_of_total_sample = noisy.shape[0]
    number_of_chunks = (number_of_total_sample + total_number_of_sample - 1) // total_number_of_sample
    number_of_total_sample_upperbound = number_of_chunks * total_number_of_sample
    padded_noisy = tf.pad(noisy, tf.constant([[0, number_of_total_sample_upperbound - number_of_total_sample], [0, 0]]),
                          "CONSTANT")
    padded_noisy = tf.reshape(padded_noisy,
                              [number_of_total_sample_upperbound // total_number_of_sample, total_number_of_sample, 1])
    estimate = model.predict(padded_noisy)
    estimate = tf.reshape(estimate, [number_of_total_sample_upperbound, 1])
    estimate = estimate[:number_of_total_sample, :]
    return tf.reshape(estimate, [-1])


def update_value_counts(value_counts, value, start=0.0020, end=0.0080, step=0.0001):
    """
    Update the dictionary with value counts categorized by specified ranges.
    """
    if value < start: return
    if value > end:
      value=end
    # Calculate the key as a string indicating the range
    # The key is determined by the floor division of the difference between the value and the start
    # divided by the step, then multiplied by the step and added back to the start
    key = start + (int((value - start) // step) * step)
    key_str = f"{key:.4f}-{key + step:.4f}"

    # Update the count for the calculated key
    if key_str not in value_counts:
        value_counts[key_str] = 1
    else:
        value_counts[key_str] += 1

def save_files(idx,loss, loss_inside_range , noisy, clean,enhanced,start=0.0020, end=0.0080, step=0.0001):

    if loss < start: return
    if loss > end:
      loss=end
    key = start + (int((loss - start) // step) * step)
    key_str = f"{key:.4f}-{key + step:.4f}"

    base_folder =  "/home/ml-dev/Data/Testing_noise_reducer/dns_2020_test_data/loss_rel/"
    noisy_folder = os.path.join(base_folder, key_str, "noisy")
    clean_folder = os.path.join(base_folder, key_str, "clean")
    enhanced_folder = os.path.join(base_folder, key_str, "enhanced")

    os.makedirs(noisy_folder, exist_ok=True)
    os.makedirs(clean_folder, exist_ok=True)
    os.makedirs(enhanced_folder, exist_ok=True)

    if loss_inside_range[key_str] <= 10:
        noisy_path = os.path.join(noisy_folder, str(loss)+"_" + str(idx) + ".wav")
        clean_path = os.path.join(clean_folder,  str(loss) + "_" + str(idx) + ".wav")
        enhanced_path = os.path.join(enhanced_folder, str(loss) + "_" + str(idx) + ".wav")
        tf.io.write_file(noisy_path , tf.audio.encode_wav(noisy, 16000))
        tf.io.write_file(clean_path, tf.audio.encode_wav(clean, 16000))
        tf.io.write_file(enhanced_path,
                         tf.audio.encode_wav(tf.reshape(enhanced, [-1, 1]), 16000))




def find_audio_high_loss(args, tfrecords_dir='/content/sample_data/tfrecords', split='train',
                         sample_rate=16000, duration=10, AUTOTUNE=tf.data.experimental.AUTOTUNE, model=''):
    if split not in ('train', 'test', 'validate'):
        raise ValueError("split must be either 'train', 'test' or 'validate'")

    # List all *.tfrecord files for the selected split
    pattern = os.path.join(tfrecords_dir, '{}*.tfrecord'.format(split))
    files_ds = tf.data.Dataset.list_files(pattern)

    # Disregard data order in favor of reading speed
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    files_ds = files_ds.with_options(ignore_order)

    # Read TFRecord files in an interleaved order
    ds = tf.data.TFRecordDataset(files_ds, compression_type='ZLIB', num_parallel_reads=AUTOTUNE)

    # Create an iterator for serial access
    iterator = iter(ds)

    filename = '/home/ml-dev/Noise_reducer/out/loss_count/loss_in_range.txt'
    loss_inside_range = {}
    i = 0
    # Loop through each batch serially
    for batch in iterator:
        # Process the batch using _parse_batch or other logic
        noisy, clean = _parse_batch(batch, sample_rate, duration, split)
        enhanced = pred(model, noisy)
        loss = mean_absolute_error_loss(clean, enhanced)

        update_value_counts(loss_inside_range, loss)
        i += 1
        loss = loss.numpy()
        #save_files(i,loss,loss_inside_range, noisy, clean, enhanced)
        if i%100 == 0:
            with open(filename, 'a') as file:
                file.write(f"Till number of examples {i}\n")
                for key in sorted(loss_inside_range.keys(), key=lambda x: float(x.split('-')[0].strip())):
                    count = loss_inside_range[key]
                    file.write(f"{key}: {count}\n")
    return None


def main(args):
    self_sample_rate = 16_000
    total_seconds_of_audio = 10
    total_number_of_sample = self_sample_rate * total_seconds_of_audio
    model = Demucs(input_shape=(total_number_of_sample, 1))
    model.load_weights(args.model_path)

    find_audio_high_loss(args, tfrecords_dir=args.tfrecord_path, model=model)
    # print(enhancer.progress)


if __name__ == '__main__':
    args = parse_args()
    main(args)

# python3 find_big_loss_file.py --model_path /home/ml-dev/Noise_reducer/out/checkpoint_epoch-007_loss-0.002056_val_loss-0.002355.h5  --tfrecord_path /media/ml-dev/1TB_volume/tfrecords_22_nov_2023/
