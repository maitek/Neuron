import os
import random
import shutil
import getopt
import argparse

parser = argparse.ArgumentParser(description='Split data set into train and test.')
parser.add_argument('input_folder',help='input folder')
parser.add_argument('output_folder',help='output folder')
parser.add_argument('--test_size',help='size of test data (between 0 and 1)', default=0.2)
parser.add_argument('--random_seed',help='seed for random shuffle', default=12345)

args = parser.parse_args()

input_folder = args.input_folder
output_folder = args.output_folder
test_size = args.test_size
random_seed = args.random_seed

splits = {"train": 1.0-test_size, "test": test_size}

# set random seed for reproducibility
random.seed(random_seed)

# list all class folders
folder_list = [file for file in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, file))]

for folder in folder_list:
	# generate split
	split_list = list()

	# get list of images
	file_list = [file for file in os.listdir(os.path.join(input_folder,folder))
				 if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
	sorted(file_list)

	num_files = len(file_list)

	num_samples_train = int(splits["train"] * num_files)
	num_samples_test = num_files - num_samples_train

	[split_list.append('train') for x in range(0,num_samples_train)]
	[split_list.append('test') for x in range(0,num_samples_test)]
	
	train_folder = os.path.join(output_folder,'train')
	test_folder = os.path.join(output_folder,'test')

	# copy files
	for idx, file in enumerate(file_list):
		dst_folder = os.path.join(output_folder,split_list[idx],folder)
		if not os.path.exists(dst_folder):
			os.makedirs(dst_folder)
		dst_file = os.path.join(dst_folder,file)
		src_file = os.path.join(input_folder,folder,file)
		shutil.copyfile(src_file,dst_file)
		print(dst_file)

	







