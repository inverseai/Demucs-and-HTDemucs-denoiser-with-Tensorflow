import tensorflow as tf
import os

def is_tfrecord_corrupted(tfrecord_path):
    try:
        sample_rate= 16000
        duration = 10
        n_samples = sample_rate * duration
        feature_description = {
            'noisy': tf.io.FixedLenFeature([n_samples], tf.float32),
            'clean': tf.io.FixedLenFeature([n_samples], tf.float32),
        }
   
        dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='ZLIB')
       
        for record in dataset:
            # Attempt to parse each record to check for corruption
            tf.io.parse_example(record, feature_description)
        #print("ok")
        return False  # No corruption detected
    except Exception as e:
        print(f"Corruption detected in {tfrecord_path}: {e}")
        return True

def find_corrupted_tfrecords(tfrecord_folder):
    corrupted_records = []
    for root, _, files in os.walk(tfrecord_folder):
        for file in files:
            if file.endswith(".tfrecord"):
                tfrecord_path = os.path.join(root, file)
                if is_tfrecord_corrupted(tfrecord_path):
                    corrupted_records.append(tfrecord_path)
    return corrupted_records

tfrecord_folder = "/media/ml-dev/1TB_volume/tfrecords_22_march/"
corrupted_records = find_corrupted_tfrecords(tfrecord_folder)

print(len(corrupted_records))
if corrupted_records:
    print("Corrupted TFRecords:")
    for record in corrupted_records:
        print(record)
else:
    print("No corrupted TFRecords found.")
