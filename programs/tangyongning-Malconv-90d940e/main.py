import torch
import pandas as pd
from dataset import ExeDataset, init_loader
from model import MalConv, MalLSTM, CNN_LSTM
from train import train_model
from sklearn.model_selection import train_test_split as spl
import os
import sys
import warnings

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

	#LOG: Best performance achieved at 99.6% acc w/ 50.7% F1 with:
	# win_size = 256
	# stride = 256
	# test_set_size = 0.25 -- and retest w/ = 1
	# mal_benign_ratio = 0.5 (by heuristic)
	# embed = 32
	# mode = "MalLSTM"
	# ON/IN: 12-2-24

	window_size = 256
	stride = window_size
	test_set_size = 0.25
	mal_benign_ratio = 0.5 #1 == all malware; 0 == all benign
	embed = 32
	mode = "mallstm"

	#dataset = 1
	log = None

	if len(sys.argv) == 7:
		window_size = int(sys.argv[1])
		stride = int(sys.argv[2])
		test_set_size = float(sys.argv[3])
		mal_benign_ratio = float(sys.argv[4])
		embed = int(sys.argv[5])
		mode = sys.argv[6]

		pth_start = './gridsearch_malratio_12-4-24/'
		multi_train = False

		if multi_train:
			model_path = pth_start+'malconv_model_'+str(sys.argv)+'_mlionestest.pth'
			optimizer_path = pth_start+'optimizer_state_'+str(sys.argv)+'_mlionestest.pth'
		log = open(pth_start+str(sys.argv)+"_LOG.txt", "w")
	
	# -----------------

	data_path = '../../../alldata_files/'
	label_path = '../../data/data.csv'

	# -----------------

	#assemble data labels as pandas table
	label_table = pd.read_csv(label_path, header = None, index_col = 0).rename(columns={1: 'ground_truth'}).groupby(level=0).last()

	#reformat table based on malware-benign ratio:
	#ASSERT(type 0 == benign and type 1 == malware)
	mal = label_table[label_table["ground_truth"] == 1]
	ben = label_table[label_table["ground_truth"] == 0]

	c_mal = len(mal)
	c_ben = len(ben)
	print("Available Malware in dataset: " + str(c_mal) + "\n"
		  + "Available Benign in dataset: " + str(c_ben) + "\n")

	if 1 > mal_benign_ratio > 0:
		#loop to determine set size
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
		else: #if the categories are of equal size
			c_mal = (c_mal + c_ben) * mal_benign_ratio
			c_ben = (c_mal + c_ben) * (1 - mal_benign_ratio)

		assert (not (c_mal == 0 or c_ben == 0))

		#take random samples of the appropriate sizes from each category and concatenate them
		label_table = pd.concat([mal.sample(int(c_mal) - 1), ben.sample(int(c_ben) - 1)])

	elif mal_benign_ratio == 1:
		label_table = label_table[label_table["ground_truth"] == 1]
	elif mal_benign_ratio == 0:
		label_table = label_table[label_table["ground_truth"] == 0]
	else:
		print("Error in malware-benign ratio parameter: proceeding with train/test using dataset standard")


	#split data
	if test_set_size == 1:
		tr_table, val_table = spl(label_table, test_size=len(label_table) - 10)
		tr_table = tr_table.sample(frac=set_size)
		val_table = val_table.sample(frac=set_size)
	else:
		tr_table, val_table = spl(label_table, test_size = test_set_size)
		tr_table = tr_table.sample(frac = set_size)
		val_table = val_table.sample(frac = set_size)
	
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
		model = MalConv(input_length=first_n_byte, window_size=window_size, stride = stride, embed = embed)
	elif mode.lower() == "mallstm":
		# CNN-LSTM MalConv
		model = MalLSTM(input_length=first_n_byte, window_size=window_size, stride = stride, embed = embed)
	elif mode.lower() == "cnn_lstm":
		model = CNN_LSTM(embed_dim=embed, device=device)
	else:
		print("MODEL LOADING ERROR\nNo Such Model '"+mode+"'")
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

	if os.path.exists(model_path) and os.path.exists(optimizer_path):
		print("Loading saved model and optimizer state...")
		model.load_state_dict(torch.load(model_path))
		optimizer.load_state_dict(torch.load(optimizer_path))
	else:
		print("No saved model found, training from scratch...")

	# ---------------

	try:
		best_model = train_model(model, criterion, optimizer, device, epochs, train_loader, valid_loader, log = log)
	except KeyboardInterrupt:
		print("interrupted")
	except RuntimeError as e:
		print(str(e))
		#print("\n!!!!!!!\nSWITCH TO CPU\n!!!!!!!\n")
		#device = torch.device('cpu')

		#criterion = torch.nn.BCEWithLogitsLoss()
		#optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

		#if mode.lower() == "cnn_lstm":
		#	model = CNN_LSTM(embed_dim=embed, device=device)

		#model = model.to(device)
		#best_model = train_model(model, criterion, optimizer, device, epochs, train_loader, valid_loader, log=log)
	finally:
		if log:
			log.close()

		if not test_set_size == 1:
			torch.save(model.state_dict(), model_path)
			torch.save(optimizer.state_dict(), optimizer_path)

		print("Model and optimizer state saved.")
