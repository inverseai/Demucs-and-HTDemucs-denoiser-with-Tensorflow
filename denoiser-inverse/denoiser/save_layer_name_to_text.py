from demucs import Demucs


model = Demucs(input_shape=(160000, 1))
model.load_weights('/home/bisnu/Training_models_Noise_reducer/without_diffq/checkpoints/checkpoint_epoch-001_loss-0.023840.h5')


names = []

for layer in model.layers:
        if "encode" in layer.name or "decode" in layer.name or "lstm":
            for param in layer.trainable_weights:
                if "kernel" in param.name:
                    names.append(layer.name)
                    s = param.name.split("/")
                    names.append(s[0])
                    names.append(s[1])
                    if layer.name == 'sequential':
                        names.append(s[2])
                elif "bias" in param.name:
                    if layer == "lstm":
                        print(param)
                    names.append(layer.name)
                    s = param.name.split("/")
                    names.append(s[0])
                    names.append(s[1])
                    if layer.name == 'sequential':
                        names.append(s[2])

print(names)