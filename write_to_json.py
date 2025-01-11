# import os
# import glob
# import json

# # Directory where the .wav files are stored
# directory_path = '/home/ml-dev/Data/DNS_dataset/mini_dataset/10_data/clean/'

# # List all .wav files in the directory
# file_paths = glob.glob(os.path.join(directory_path, '*.wav')) + glob.glob(os.path.join(directory_path, '*.flac'))
# sample_rate = 48000
# duration = 10 
# total_sample = sample_rate * duration

# # Prepare the list to be written to the JSON file, including the total_sample (160000) for each file
# file_info = [[file_path, total_sample] for file_path in file_paths]

# # JSON file to write the output to
# output_json_path = '/home/ml-dev/Data/DNS_dataset/mini_dataset/10_data/clean.json'

# # Write the list to the JSON file
# with open(output_json_path, 'w') as json_file:
#     json.dump(file_info, json_file, indent=4)

# print(f"File list written to {output_json_path}")


import os
import glob
import json

def create_file_list(directory_path, sample_rate, duration):
    """
    Generates a list of audio file paths and their total sample counts.

    Args:
        directory_path (str): The directory containing audio files.
        sample_rate (int): The sample rate of the audio files.
        duration (int): The duration of the audio files in seconds.

    Returns:
        list: A list of [file_path, total_samples] pairs.
    """
    # List all .wav and .flac files in the directory
    file_patterns = ('*.wav', '*.flac')
    file_paths = []
    for pattern in file_patterns:
        file_paths.extend(glob.glob(os.path.join(directory_path, pattern)))
    
    total_samples = sample_rate * duration
    
    # Prepare the list to be written to the JSON file
    file_info = [[file_path, total_samples] for file_path in file_paths]
    return file_info

def write_to_json(data, output_path):
    """
    Writes data to a JSON file.

    Args:
        data (list): The data to write to the JSON file.
        output_path (str): The path to the output JSON file.
    """
    with open(output_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    print(f"File list written to {output_path}")

def main():
    # Base directory path
    base_path = '/home/ml-dev/Data/mini_dataset/10_data_wav'
    # Folders to process
    folders = ['clean', 'noisy']
    # Audio parameters
    sample_rate = 48000
    duration = 10  # in seconds

    for folder in folders:
        directory_path = os.path.join(base_path, folder)
        output_json_path = os.path.join(base_path, f'{folder}.json')
        
        # Generate the file list
        file_info = create_file_list(directory_path, sample_rate, duration)
        # Write the file list to a JSON file
        write_to_json(file_info, output_json_path)

if __name__ == '__main__':
    main()

