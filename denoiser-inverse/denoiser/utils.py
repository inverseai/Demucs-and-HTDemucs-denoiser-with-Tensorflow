import tensorflow as tf
import os
import json

def save_model(model, path):
	print(f'the model is saving into this path: {path}')
	model.save(path)

def write_history(history, history_json_location):
	if os.path.isfile(history_json_location):
		with open(history_json_location, 'r+', encoding ='utf8') as json_file:
			data = json.load(json_file)
			for key in history.history.keys():
				data[key] += history.history[key]
			json_file.seek(0) # <--- should reset file position to the beginning.
			json.dump(data, json_file, indent=4)
			json_file.truncate()    
	else:
		with open(history_json_location, 'w', encoding ='utf8') as json_file:
			json.dump(history.history, json_file, indent = 4)
