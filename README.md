# Demucs-and-HTDemucs-based-denoiser-with-Tensorflow


## Description

We have implemented state-of-the-art Demucs and HTDemucs architecture for noise reduction in tensorflow.
We converted Pytorch implementation from [facebook’s Demucs Speech Enhancement](https://github.com/facebookresearch/denoiser) and [Hybrid Transformers for Music Source Separation](https://arxiv.org/abs/2211.08553)(The model, originally designed for music source separation, was updated by our team to perform noise reduction) for our convenient deployment purposes.

In the original work, they provide the description as,  
“We provide a causal speech enhancement model working on the raw waveform that runs in real-time on a laptop CPU. The proposed model is based on an encoder-decoder architecture with skip connections. It is optimized on both time and frequency domains, using multiple loss functions. Empirical evidence shows that it is capable of removing various kinds of background noise including stationary and non-stationary noises, as well as room reverb. Additionally, we suggest a set of data augmentation techniques applied directly on the raw waveform which further improve model performance and its generalization abilities.”

We weren’t able to fully replicate their architecture. In the loss function, we currently use waveform loss only and also we implemented causal speech enhancement(only based on current and past samples)


## Installation

Install python3 with conda environment
```
conda create -n <env_name> python=3.9
```

Activate conda environment
```
conda activate <env_name>
```

If you just want to run on GPU setup follow Guides provided for GPU setup below. Otherwise you can install required packages through :
```
cd noise-reducer-ml
pip install -r requirements.txt
```


## GPU Setup for TensorFlow Training on Ubuntu 22.04

This guide outlines the steps to configure a GPU for training deep learning models with TensorFlow on an Ubuntu 22.04 system.

**Prerequisites:**

* Ubuntu 22.04
* NVIDIA GPU with compatible drivers (>= 470.161.03)

**Installation:**

1. **NVIDIA Drivers:**
    - Follow method 2 from the guide "Install Nvidia Drivers on Ubuntu 20.04 {3 Methods}": https://ubuntu.com/server/docs/nvidia-drivers-installation
2. **TensorFlow and cuda installation:**
    - Follow the official TensorFlow installation instructions for pip installation: https://www.tensorflow.org/install/pip

**Troubleshooting:**

- **`libnvinfer.so.7` not found:**
    - Follow th instruction (https://github.com/tensorflow/tensorflow/issues/57679).
- **`Can't find libdevice directory`:**
    - Follow solution:  (https://github.com/tensorflow/tensorflow/issues/58681#issuecomment-1371317455)
- **`Couldn't invoke ptxas --version`:**
    - Follow solution: (https://discuss.tensorflow.org/t/cant-find-libdevice-directory-cuda-dir-nvvm-libdevice/11896/7)



## Data preparation

- Download dataset from GitHub - microsoft/DNS-Challenge at interspeech2020/master repository or download links provided in the directory Dataset_preparation/Dataset_preparation_required_files/
-  Extract data from tar files (scripts contain in Dataset_preparation/Dataset_preparation_required_files/)

#### Generate noisy and clean audio files 
  -  cd to Dataset_preparation/DNS-Challenge_2020_setup.
  -  Install required files from requirements.txt (pip install -r requirements.txt)
  -  Setup parameters noisyspeech_synthesizer.cfg.
  -  Generate noisy and clean audio files by running script noisyspeech_synthesizer_singleprocess.py.
 
#### Use raw data or generate Tfrecords from noisy and clean audios
- Change directory to denoiser_inverse/denoiser
- Write file path to a json file with running convert.py

```
python3 convert.py --noisy_dir <noisy files path> --clean_dir <clean_files_path> --noisy_json_location <noisy json file location(with .json extension)> --clean_json_location <clean json file location(with .json extension)>
```
- Generate tfrecord files with running convert_tf_record.py
```
python3 convert_tf_record.py -n <noisy json file location> -c <clean json file location> -o  <tfrecord path> --num-shards-train <number of train tfrecords> --num-shards-test <number of test tfrecords> --num-shards-val <number of valid tfrecords> --sample-rate 16000 --test-size <test size in percentage(values between 0 to 1)> --val-size <validation size in percentage(values between 0 to 1)>
```

#### Create json file of train and validation noisy and clean files
- Create json file using write_to_json.py for writing audio paths in json files(updates data path in write_to_json)


#### Troubleshooting
In case of getting tfrecord corruption issue(DATA_LOSS:  corrupted record), We can figure out the tfrecords which contains issues .
- Change directory to Train/denoiser_inverse and run find_corrupted_tfrecords.py(Update tfrecord path in the script)

## Training

- Change directory to denoiser_inverse
- update data path and configurations for training in config.yaml
- Run train_main.py for training. 
```
python3 train_main.py  
```

***
# Evaluation
- Install packages for evaluation:
```
pip3 install pystoi
pip3 install pesq
```
- Change directory to Train/denoiser_inverse
- Run evl.py for evaluation 
```
python3 evl.py --model_path <model_path> --tfrecord_path  <tfrecord path for test datasets>
```



***
# Inference

- Change directory to Train/denoiser_inverse/denoiser
- Run enhance_with_model.py for inference: 
```
python3 inference_main.py --checkpoint_path  <model_path> --input_dir <noisy_path> --output_dir <enhanced_path>
```


***
# Benchmark of our trained and deployed Model 

| Model Name                   | Dataset Name                                   | Dataset Duration | Dataset Updates                                            | Parameters | MOS Evaluation |
|------------------------------|------------------------------------------------|------------------|------------------------------------------------------------|------------|----------------|
| RNNoise_0.1 |                                               |                  |                                                            |            | 3.2024         |
| RNNoise_0.2                  |                                                |                  |                                                            |            | 3.4372         |
| Demucs                       | Deep Noise Suppression (DNS) Challenge 4 - ICASSP 2022 | 1000h           | Cleaning voices from noise audios and cleaning non-voice portions from speech audio | 32 Million | 3.6085         |
| HTDemucs                     | Deep Noise Suppression (DNS) Challenge 4 - ICASSP 2022 | 1000h           | Cleaning voices from noise audios and cleaning non-voice portions from speech audio | 26 Million | 3.7748         |


