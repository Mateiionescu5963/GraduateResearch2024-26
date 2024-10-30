import torch
import pandas as pd
from dataset import ExeDataset, init_loader
from model import MalConv
from train import train_model
from sklearn.model_selection import train_test_split as spl
import os
import warnings

if __name__ == "__main__":
	warnings.filterwarnings("ignore")

	data_path = '../../../MicrosoftData2015/train/'
	label_path = '../../data/MicrosoftData2015/trainLabels_mod.csv'
	
	# -- 
	model_path = 'malconv_model_mlionestest.pth'
	optimizer_path = 'optimizer_state_mlionestest.pth'
	# -- 
	first_n_byte = 2000000
	window_size = 500
	stride = window_size
	test_set_size = 0.25
	batch_size = 1
	epochs = 10
	
	# -----------------

	#assemble data labels as pandas table
	label_table = pd.read_csv(label_path, header = None, index_col = 0).rename(columns={1: 'ground_truth'}).groupby(level=0).last()
	
	#split data 
	tr_table, val_table = spl(label_table, test_size = test_set_size)
	
	# --------------

	print('Training Set:')
	print('\tTotal', len(tr_table), 'files')
	#print('\tMalware Count :', tr_table['ground_truth'].value_counts().iloc[1])
	#print('\tGoodware Count:', tr_table['ground_truth'].value_counts().iloc[0])

	print('Validation Set:')
	print('\tTotal', len(val_table), 'files')
	#print('\tMalware Count :', val_table['ground_truth'].value_counts().iloc[1])
	#print('\tGoodware Count:', val_table['ground_truth'].value_counts().iloc[0])
	
	# --------------

	train_dataset = ExeDataset(list(tr_table.index), data_path, list(tr_table.ground_truth), first_n_byte)
	valid_dataset = ExeDataset(list(val_table.index), data_path, list(val_table.ground_truth), first_n_byte)

	train_loader, valid_loader = init_loader(train_dataset, batch_size)
	valid_loader = init_loader(valid_dataset, batch_size)[1]

	model = MalConv(input_length=first_n_byte, window_size=window_size)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = model.to(device)
	
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
		best_model = train_model(model, criterion, optimizer, device, epochs, train_loader, valid_loader)
	except KeyboardInterrupt:
		print("interrupted")
	finally:
		torch.save(model.state_dict(), model_path)
		torch.save(optimizer.state_dict(), optimizer_path)

		print("Model and optimizer state saved.")
