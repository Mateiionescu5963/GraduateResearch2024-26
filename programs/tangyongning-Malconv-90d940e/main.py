import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from dataset import ExeDataset, init_loader
from model import MalConv, MalLSTM, CNN_LSTM, MalTF
from train import train_model, eval_model
from analysis import extract_scores
from sklearn.model_selection import train_test_split as spl
import os
import sys
import warnings

def balance_dataset(path, mal_benign_ratio):
	# assemble data labels as pandas table
	label_table = pd.read_csv(path, header=None, index_col=0).rename(columns={1: 'ground_truth'}).groupby(level=0).last()

	# reformat table based on malware-benign ratio:
	# ASSERT(type 0 == benign and type 1 == malware)
	mal = label_table[label_table["ground_truth"] == 1]
	ben = label_table[label_table["ground_truth"] == 0]

	c_mal = len(mal)
	c_ben = len(ben)
	print("Available Malware in dataset: " + str(c_mal) + "\n"
		  + "Available Benign in dataset: " + str(c_ben) + "\n")

	if 1 > mal_benign_ratio > 0:
		# loop to determine set size
		if not c_mal == c_ben:
			if not mal_benign_ratio == 0.5:
				if c_mal / (c_mal + c_ben) > mal_benign_ratio:
					while not abs((c_mal / (c_mal + c_ben)) - mal_benign_ratio) < 2 / (c_mal + c_ben):
						c_mal -= 1
						if not c_mal / (c_mal + c_ben) > mal_benign_ratio:
							break
				else:
					while not abs((c_ben / (c_mal + c_ben)) - mal_benign_ratio) < 2 / (c_mal + c_ben):
						c_ben -= 1
						if c_mal / (c_mal + c_ben) > mal_benign_ratio:
							break
			else:
				c_mal = c_ben
		else:  # if the categories are of equal size
			c_mal = (c_mal + c_ben) * mal_benign_ratio
			c_ben = (c_mal + c_ben) * (1 - mal_benign_ratio)

		assert (not (c_mal == 0 or c_ben == 0))

		# take random samples of the appropriate sizes from each category and concatenate them
		return pd.concat([mal.sample(int(c_mal) - 1), ben.sample(int(c_ben) - 1)])

	elif mal_benign_ratio == 1:
		return label_table[label_table["ground_truth"] == 1]
	elif mal_benign_ratio == 0:
		return label_table[label_table["ground_truth"] == 0]
	else:
		print("Error in malware-benign ratio parameter: proceeding with train/test using dataset standard")
		return label_table

def train(model_path, optimizer_path, first_n_byte, set_size, batch_size, epochs, window_size, stride, test_set_size, embed, mode, log, label_table, dataset_test = False, tr_table = pd.DataFrame(), val_table = pd.DataFrame()):
	# split data
	if tr_table.empty or val_table.empty:
		if test_set_size == 1:
			tr_table, val_table = spl(label_table, test_size=len(label_table) - 10)
			tr_table = tr_table.sample(frac=set_size)
			val_table = val_table.sample(frac=set_size)
		else:
			tr_table, val_table = spl(label_table, test_size=test_set_size)
			tr_table = tr_table.sample(frac=set_size)
			val_table = val_table.sample(frac=set_size)

	# --------------

	print('Training Set:')
	print('\tTotal', len(tr_table), 'files')
	print('\tMalware Count :', len(tr_table[tr_table['ground_truth'] == 1]))
	print('\tGoodware Count:', len(tr_table[tr_table['ground_truth'] == 0]))

	print('Validation Set:')
	print('\tTotal', len(val_table), 'files')
	print('\tMalware Count :', len(val_table[val_table['ground_truth'] == 1]))
	print('\tGoodware Count:', len(val_table[val_table['ground_truth'] == 0]))

	# --------------

	# dataset files as dataset.py objects
	train_dataset = ExeDataset(list(tr_table.index), data_path, list(tr_table.ground_truth), first_n_byte)
	valid_dataset = ExeDataset(list(val_table.index), data_path, list(val_table.ground_truth), first_n_byte)

	# datasets as pytorch utils objects
	train_loader, valid_loader = init_loader(train_dataset, batch_size)
	valid_loader = init_loader(valid_dataset, batch_size)[1]

	# set device to cuda GPU if available
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# load model format
	if mode.lower() == "malconv":
		# Standard Malconv
		model = MalConv(input_length=first_n_byte, window_size=window_size, stride=stride, embed=embed)
	elif mode.lower() == "mallstm":
		# CNN-LSTM MalConv
		model = MalLSTM(input_length=first_n_byte, window_size=window_size, stride=stride, embed=embed)
	elif mode.lower() == "cnn_lstm":
		model = CNN_LSTM(embed_dim=embed, device=device)
	elif mode.lower() == "maltf":
		model = MalTF(input_length=first_n_byte, window_size=window_size, stride=stride, embed=embed)
	else:
		print("MODEL LOADING ERROR\nNo Such Model '" + mode + "'")
		exit(1)

	model = model.to(device)

	# set pytorch loss and optimization parameters
	criterion = torch.nn.BCEWithLogitsLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

	# --------------

	if torch.cuda.is_available():
		print("CUDA is available. Training on GPU.")
	else:
		print("CUDA is not available. Training on CPU.")

	if not dataset_test:
		if os.path.exists(model_path) and os.path.exists(optimizer_path):
			print("Loading saved model and optimizer state...")
			model.load_state_dict(torch.load(model_path))
			optimizer.load_state_dict(torch.load(optimizer_path))
		else:
			print("No saved model found, training from scratch...")

	# ---------------

	try:
		best_model = train_model(model, criterion, optimizer, device, epochs, train_loader, valid_loader, log=log)
		if not test_set_size == 1 and not dataset_test:
			torch.save(model.state_dict(), model_path)
			torch.save(optimizer.state_dict(), optimizer_path)

			print("Model and optimizer state saved.")
	except KeyboardInterrupt:
		print("interrupted")
		if not test_set_size == 1 and not dataset_test:
			torch.save(model.state_dict(), model_path)
			torch.save(optimizer.state_dict(), optimizer_path)

			print("Model and optimizer state saved.")
	except RuntimeError as e:
		print(str(e))
	# print("\n!!!!!!!\nSWITCH TO CPU\n!!!!!!!\n")
	# device = torch.device('cpu')

	# criterion = torch.nn.BCEWithLogitsLoss()
	# optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

	# if mode.lower() == "cnn_lstm":
	#	model = CNN_LSTM(embed_dim=embed, device=device)

	# model = model.to(device)
	# best_model = train_model(model, criterion, optimizer, device, epochs, train_loader, valid_loader, log=log)
	finally:
		if log:
			log.close()
	return model

if __name__ == "__main__":
	warnings.filterwarnings("ignore")

	# --
	model_path = 'malconv_model_mlionestest.pth'
	optimizer_path = 'optimizer_state_mlionestest.pth'
	# --

	first_n_byte = 2000000
	set_size = 1.0
	batch_size = 1
	epochs = 10

	#LOG: Best performance achieved at 99.9% acc w/ 50.8% F1 with:
	# win_size = 128
	# stride = 64
	# test_set_size = 0.25 -- and retest at 1.0 gives 99.5% accuracy w/ 49.4% F1
	# mal_benign_ratio = 0.5 (by heuristic)
	# embed = 24
	# mode = "MalTF"
	# ON/IN: 2-4-25

	#log best MalLSTM hyperparam:
	# size = 256
	# stride = 256
	# test_set = .25
	# mal_benign_ratio = 0.5
	# embed = 32

	window_size = 128
	stride = 64
	test_set_size = 0.25
	mal_benign_ratio = 0.5 #1 == all malware; 0 == all benign
	embed = 24
	mode = "maltf"

	#dataset = 1
	log = None
	dataset_test = False
	corruption = False

	if len(sys.argv) == 3:
		dataset_test = bool(sys.argv[1])
		corruption = bool(sys.argv[2])
	elif len(sys.argv) >= 7:
		window_size = int(sys.argv[1])
		stride = int(sys.argv[2])
		test_set_size = float(sys.argv[3])
		if test_set_size == 1.0:
			sys.argv[3] = "0.25"
		mal_benign_ratio = float(sys.argv[4])
		embed = int(sys.argv[5])
		mode = sys.argv[6]

		pth_start = './maltf-gridsearch_2-4-25/'
		multi_train = False
		if len(sys.argv) >= 8:
			multi_train = bool(sys.argv[7])

			sys.argv = sys.argv[:7]

		if multi_train:
			model_path = pth_start+'malconv_model_'+str(sys.argv)+'_mlionestest.pth'
			optimizer_path = pth_start+'optimizer_state_'+str(sys.argv)+'_mlionestest.pth'
		log = open(pth_start+str(sys.argv)+"_LOG.txt", "w")
	
	# -----------------

	data_path = '../../../alldata_files/'
	label_path = '../../data/data.csv'

	# -----------------

	label_table = balance_dataset(label_path, mal_benign_ratio)


	if dataset_test:
		#test_set_size = 0.5
		#set_size = 0.2

		test_table, validation_table = spl(label_table, test_size=test_set_size)
		test_table = test_table.sample(frac=set_size)
		validation_table = validation_table.sample(frac=set_size)

		model = train(model_path, optimizer_path, first_n_byte, set_size, batch_size, epochs, window_size, stride, test_set_size, embed, mode, log, label_table, dataset_test = True, tr_table = test_table, val_table = validation_table)
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		dataset_test_results = pd.DataFrame()
		try:
			dataset_test_results = pd.read_csv("./ds_tst.csv", index_col = 0)
			dataset_test_results.to_csv("./ds_tst_bak.csv")
		except FileNotFoundError:
			print("Dataset Testing First Initialization")

		if dataset_test_results.empty:
			dataset_test_results = pd.DataFrame(columns=["Name", "Label", "Corrupted", "Accuracies", "Trials"])
			dataset_test_results.set_index("Name", inplace = True)

		try:
			corrupt_table = pd.DataFrame()

			if corruption:
				corrupt_table = pd.read_csv("./corruption.csv", index_col=0)

			for index, row in validation_table.iterrows():
				valid_loader = None
				corrupt = False

				if corruption:
					if index in corrupt_table.index:
						corrupt = True
						valid_sample = ExeDataset([index], data_path, [1 if row['ground_truth'] == 0 else 0], first_n_byte)
						valid_loader = DataLoader(valid_sample, batch_size=batch_size, drop_last=False)
					else:
						valid_sample = ExeDataset([index], data_path, [row['ground_truth']], first_n_byte)
						valid_loader = DataLoader(valid_sample, batch_size=batch_size,drop_last=False)

				acc, pre, rec, f1 = eval_model(model, valid_loader, device)
				if acc == 0:
					acc = -1

				if index not in dataset_test_results.index:
					dataset_test_results.loc[index] = [row['ground_truth'], corrupt, acc, 1]
				else:
					accuracies = dataset_test_results.at[index, "Accuracies"]
					dataset_test_results.at[index, "Accuracies"] = accuracies + acc
					trials = dataset_test_results.at[index, "Trials"]
					dataset_test_results.at[index, "Trials"] = trials + 1
					dataset_test_results.at[index, "Corrupted"] = corrupt

			dataset_test_results.sort_values(by=["Trials"], ascending=False)
			dataset_test_results.to_csv("./ds_tst.csv")
		except KeyboardInterrupt:
			print("Interrupt")
		#finally:
		#	dataset_test_results.sort_values(by = ["Trials"], ascending = False)
		#	dataset_test_results.to_csv("./ds_tst.csv")

		# #first pass shapely
		# exclusion_threshold = -0.005
		# excluded_set_indices = []
		#
		# log = open("temp.txt", "w")
		# train(model_path, optimizer_path, first_n_byte, set_size, batch_size, epochs, window_size, stride, test_set_size, embed, mode, log, label_table, dataset_test = True)
		# init_results = extract_scores("temp.txt")
		#
		# #shuffle dataset
		# label_table = label_table.sample(frac = 1)
		#
		# #create a number of subsets equal to the sqrt of the full dataset size
		# n = len(label_table)
		# subset_size = int(np.sqrt(n))
		# print("Subset Size is: "+str(subset_size))
		#
		# label_sets = np.array_split(label_table, subset_size)
		#
		# #for each set, exclude it and train
		# try:
		# 	for i, excluded in enumerate(label_sets):
		# 		test_labels = label_table[~label_table.isin(excluded)].dropna()
		#
		# 		log.open("temp.txt", "w")
		# 		train(model_path, optimizer_path, first_n_byte, set_size, batch_size, epochs, window_size, stride, test_set_size, embed, mode, log, test_labels, dataset_test=True)
		#
		# 		results = extract_scores("temp.txt")
		#
		# 		change = init_results[0] - results[0]
		# 		if change > 0 or change >= exclusion_threshold:
		# 			excluded_set_indices.append(i)
		# except Exception as e:
		# 	print("An exception occurred during processing of exclusions")
		# 	print(e)
		# finally:
		# 	if log:
		# 		log.close()
		#
		# exclusion_zone = None
		# for index in excluded_set_indices:
		# 	if exclusion_zone:
		# 		exclusion_zone = pd.concat([exclusion_zone, label_sets[index]])
		# 	else:
		# 		exclusion_zone = label_sets[index]
		#
		# if exclusion_zone:
		# 	exclusion_zone.to_csv('potential_exclusion.csv', index = True)
	else:
		train(model_path, optimizer_path, first_n_byte, set_size, batch_size, epochs, window_size, stride, test_set_size, embed, mode, log, label_table)


