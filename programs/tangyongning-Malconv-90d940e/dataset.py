import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class ExeDataset(Dataset):
	def __init__(self, fp_list, data_path, label_list, first_n_byte=2000000):
		self.fp_list = fp_list
		self.data_path = data_path
		self.label_list = label_list
		self.first_n_byte = first_n_byte

	def __len__(self):
		return len(self.fp_list)

	def __getitem__(self, idx):
		try:
			# attempt to open file without file extension
			with open(self.data_path + self.fp_list[idx], 'rb') as f:
				# numpy buffer read
				tmp = np.frombuffer(f.read(self.first_n_byte), dtype=np.uint8) + 1
				if len(tmp) < self.first_n_byte:
					tmp = np.pad(tmp, (0, self.first_n_byte - len(tmp)), 'constant')
		except FileNotFoundError:
			try:
				# as above but using specific extensions when file is not found
				with open(self.data_path + self.fp_list[idx] + ".exe", 'rb') as f:
					tmp = np.frombuffer(f.read(self.first_n_byte), dtype=np.uint8) + 1
					if len(tmp) < self.first_n_byte:
						tmp = np.pad(tmp, (0, self.first_n_byte - len(tmp)), 'constant')
			except FileNotFoundError:
				try:
					# as above but using specific extensions when file is not found
					with open(self.data_path + self.fp_list[idx] + ".ole", 'rb') as f:
							tmp = np.frombuffer(f.read(self.first_n_byte), dtype=np.uint8) + 1
							if len(tmp) < self.first_n_byte:
								tmp = np.pad(tmp, (0, self.first_n_byte - len(tmp)), 'constant')
				except FileNotFoundError:
					try:
						# as above but using specific extensions when file is not found
						with open(self.data_path + self.fp_list[idx] + ".asm", 'rb') as f:
							tmp = np.frombuffer(f.read(self.first_n_byte), dtype=np.uint8) + 1
							if len(tmp) < self.first_n_byte:
								tmp = np.pad(tmp, (0, self.first_n_byte - len(tmp)), 'constant')
					except FileNotFoundError:
						with open(self.data_path + self.fp_list[idx] + ".bytes", 'rb') as f:
							tmp = np.frombuffer(f.read(self.first_n_byte), dtype=np.uint8) + 1
							if len(tmp) < self.first_n_byte:
								tmp = np.pad(tmp, (0, self.first_n_byte - len(tmp)), 'constant')

		return tmp, np.array([self.label_list[idx]], dtype=np.float32)

def init_loader(dataset, batch_size=8):
	train_size = int(0.8 * len(dataset))
	valid_size = len(dataset) - train_size

	train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
	valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
	return train_loader, valid_loader
