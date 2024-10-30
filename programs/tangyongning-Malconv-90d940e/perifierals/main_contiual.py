import torch
import pandas as pd
from dataset import ExeDataset, init_loader
from model_continue import MalConv
from train import train_model
import os

train_data_path = 'Malconv/data/batch_10/train/'
train_label_path = 'Malconv/data/batch_10_train.csv'
valid_data_path = 'Malconv/data/batch_10/test/'
valid_label_path = 'Malconv/data/batch_10_test.csv'

# 读取并处理标签文件
tr_label_table = pd.read_csv(train_label_path, header=None, index_col=0)
tr_label_table.index = tr_label_table.index.str.upper()
tr_label_table = tr_label_table.rename(columns={1: 'ground_truth'})
val_label_table = pd.read_csv(valid_label_path, header=None, index_col=0)
val_label_table.index = val_label_table.index.str.upper()
val_label_table = val_label_table.rename(columns={1: 'ground_truth'})

# 获取唯一的训练和验证集
tr_table = tr_label_table.groupby(level=0).last()
del tr_label_table
val_table = val_label_table.groupby(level=0).last()
del val_label_table
tr_table = tr_table.drop(val_table.index.join(tr_table.index, how='inner'))

# 打印数据集信息
print('Training Set:')
print('\tTotal', len(tr_table), 'files')
print('\tMalware Count :', tr_table['ground_truth'].value_counts().iloc[1])
print('\tGoodware Count:', tr_table['ground_truth'].value_counts().iloc[0])

print('Validation Set:')
print('\tTotal', len(val_table), 'files')
print('\tMalware Count :', val_table['ground_truth'].value_counts().iloc[1])
print('\tGoodware Count:', val_table['ground_truth'].value_counts().iloc[0])

# 参数设置
first_n_byte = 2000000
window_size = 500
batch_size = 16
epochs = 10

# 创建数据集
train_dataset = ExeDataset(list(tr_table.index), train_data_path, list(tr_table.ground_truth), first_n_byte)
valid_dataset = ExeDataset(list(val_table.index), valid_data_path, list(val_table.ground_truth), first_n_byte)

# 初始化数据加载器
train_loader, valid_loader = init_loader(train_dataset, batch_size)
valid_loader = init_loader(valid_dataset, batch_size)[1]

# 模型、损失函数和优化器
model = MalConv(input_length=first_n_byte, window_size=window_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

# 检查是否存在已保存的模型和优化器状态
model_path = 'malconv_model_CBP.pth'
optimizer_path = 'optimizer_state_CBP.pth'

if os.path.exists(model_path) and os.path.exists(optimizer_path):
    print("Loading saved model and optimizer state...")
    model.load_state_dict(torch.load(model_path))
    optimizer.load_state_dict(torch.load(optimizer_path))
else:
    print("No saved model found, training from scratch...")

# 训练模型
best_model = train_model(model, criterion, optimizer, device, epochs, train_loader, valid_loader)

# 训练完成后保存模型和优化器状态
torch.save(model.state_dict(), 'malconv_model_CBP.pth')
torch.save(optimizer.state_dict(), 'optimizer_state_CBP.pth')

print("Model and optimizer state saved.")
