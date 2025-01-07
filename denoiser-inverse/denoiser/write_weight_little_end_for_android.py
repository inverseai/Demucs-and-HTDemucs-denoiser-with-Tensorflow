from distutils.file_util import write_file
from demucs import Demucs
import tensorflow as tf
import numpy as np
import struct


model = Demucs(input_shape=(160000, 1))
model.load_weights('/home/ml-dev/Noise_reducer/out/checkpoint_25_nov/checkpoint_epoch-016_loss-0.002084.h5')


names = []
#f = open("weight_bias.txt", "a")
f = open("weight_mat_little_end_002084.bin", "wb")

for layer in model.layers:
        if "encode" in layer.name or "decode" in layer.name or "lstm":
            for param in layer.trainable_weights:
                if "kernel" in param.name:
                    weight = param.numpy()
                    shape = weight.shape
                    print(layer.name)
                    print("Weight :", shape)
                    d=struct.pack('<i',shape[0])
                    f.write(d)
                    d=struct.pack('<i',shape[1])
                    f.write(d)
                    if layer.name != 'sequential':
                        d=struct.pack('<i',shape[2])
                        f.write(d)
                        for i in range(len(weight)):
                            for j in range(len(weight[i])):
                                for k in range(len(weight[i][j])):
                                    d=struct.pack('<f',weight[i][j][k])
                                    f.write(d)
                    else:
                        for i in range(len(weight)):
                            for j in range(len(weight[i])):
                                d=struct.pack('<f',weight[i][j])
                                f.write(d)



                    #f.write("{} {} {}", shape[0],shape[1],shape[2])
            
                
                    #print(shape)
                    #print(weight)
                elif "bias" in param.name:
                    bias = param.numpy()
                    shape = bias.shape
                    print(layer.name)
                    print("Bias ", shape)
                    
                    d=struct.pack('<i',shape[0])
                    f.write(d)

                    for i in range(len(bias)):
                        d=struct.pack('<f',bias[i])
                        f.write(d)

f.close()
#print(names)