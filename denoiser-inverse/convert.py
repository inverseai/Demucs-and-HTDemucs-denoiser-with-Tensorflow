from ast import arg
import json
import logging
import os
import re
import argparse



def make_json_from_dir(dir, json_dir, memory_th, file_format = {'.wav'}):
	file_list = os.listdir(dir)
	path_list = []
	for file_name in file_list:
		if os.path.splitext(file_name)[-1] in file_format and os.path.getsize(os.path.join(dir, file_name)) >= memory_th:	
			path_list += [os.path.join(dir, file_name)]
	#print(path_list)
	with open(json_dir, 'w') as f:
		json.dump(path_list, f, indent=4)

def match_dns(noisy, clean):
	"""match_dns.
	Match noisy and clean DNS dataset filenames.
	:param noisy: list of the noisy filenames
	:param clean: list of the clean filenames
	"""
	print("Matching noisy and clean for dns dataset")
	noisydict = {}
	extra_noisy = []
	for path in noisy:
		# print(path)
		match = re.search(r'fileid_(\d+)\.wav$', path)
		if match is None:
			# maybe we are mixing some other dataset in
			extra_noisy.append(path)
		else:
			noisydict[match.group(1)] = path
	noisy[:] = []
	extra_clean = []
	copied = list(clean)
	clean[:] = []
	for path in copied:
		# print(path)
		match = re.search(r'fileid_(\d+)\.wav$', path)
		if match is None:
			extra_clean.append(path)
		else:
			if match.group(1) in noisydict:
				noisy.append(noisydict[match.group(1)])
				clean.append(path)
	extra_noisy.sort()
	extra_clean.sort()
	clean += extra_clean
	noisy += extra_noisy

def modify(noisy_locaton, clean_location):
	with open(noisy_locaton, 'r+') as ns , open(clean_location, 'r+') as cl:
		data_noisy = json.load(ns)
		data_clean = json.load(cl)
		match_dns(data_noisy, data_clean)
		ns.seek(0) # <--- should reset file position to the beginning.
		cl.seek(0) # <--- should reset file position to the beginning.
		json.dump(data_noisy, ns, indent=4)
		json.dump(data_clean, cl, indent=4)
		ns.truncate() # remove remaining part
		cl.truncate() # remove remaining part


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--noisy_dir', type=str, dest='noisy_dir',
						default= '',
						help='directory for noisy wav file'
						'(default: %(default)s)')
	parser.add_argument('--clean_dir', type=str, dest='clean_dir',
						default= '',
						help='directory for clean wav file'
						'(default: %(default)s)')
	parser.add_argument('--noisy_json_location', type=str, dest='noisy_json_location',
						default= '',
						help='lation for noisy json file'
						'(default: %(default)s)')
	parser.add_argument('--clean_json_location', type=str, dest='clean_json_location',
						default= '',
						help='location for clean json file'
						'(default: %(default)s)')
	parser.add_argument('--memory_th', type=int, dest='memory_th',
						default= '100',
						help='will not process wav if file size is smaller than memory_th bytes'
						'(default: %(default)s)')
	return parser.parse_args()

if __name__ == '__main__':
	args = parse_args()

	if args.noisy_dir != '' and args.noisy_json_location != '':
		print(args.noisy_dir, args.noisy_json_location)
		make_json_from_dir(args.noisy_dir, args.noisy_json_location, args.memory_th)
	if args.clean_dir != '' and args.clean_json_location != '':
		make_json_from_dir(args.clean_dir, args.clean_json_location, args.memory_th)
	if args.noisy_json_location != '' and args.clean_json_location != '':
		if os.path.exists(args.noisy_json_location) and os.path.exists(args.clean_json_location):
			modify(args.noisy_json_location, args.clean_json_location)
		else:
			print('noisy or clean json file not found!!!')
			exit(0)


#python3 convert.py --noisy_dir /home/ml-dev/Data/Testing_noise_reducer/dns_2020_test_data/reverb/noisy/ --clean_dir /home/ml-dev/Data/Testing_noise_reducer/dns_2020_test_data/reverb/clean/ --noisy_json_location /home/ml-dev/Data/Testing_noise_reducer/dns_2020_test_data/reverb/noisy.json --clean_json_location /home/ml-dev/Data/Testing_noise_reducer/dns_2020_test_data/reverb/clean.json