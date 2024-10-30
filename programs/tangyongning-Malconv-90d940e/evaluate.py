import torch
import pandas as pd
from dataset import ExeDataset, init_loader
from model_continue import MalConv
from sklearn.metrics import accuracy_score

# 定义评估函数
def evaluate_model(model, loader, device):
    model.eval()  # 设置模型为评估模式
    all_preds = []
    all_labels = []

    with torch.no_grad():  # 评估时不需要计算梯度
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            
            # 模型前向传播
            outputs = model(data)
            preds = torch.sigmoid(outputs) > 0.5  # 二分类预测，使用sigmoid函数并根据阈值0.5判定
            
            # 收集预测结果和标签
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算准确率
    accuracy = accuracy_score(all_labels, all_preds)
    print(f'Validation Accuracy: {accuracy:.4f}')
    return accuracy

# 参数设置
first_n_byte = 2000000
window_size = 500
batch_size = 16

# 路径设置
valid_data_path = 'Malconv/data/valid/'
valid_label_path = 'Malconv/data/valid-label.csv'

# 加载验证集
val_label_table = pd.read_csv(valid_label_path, header=None, index_col=0)
val_label_table.index = val_label_table.index.str.upper()
val_label_table = val_label_table.rename(columns={1: 'ground_truth'})

val_table = val_label_table.groupby(level=0).last()
del val_label_table

# 打印验证集信息
print('Validation Set:')
print('\tTotal', len(val_table), 'files')
print('\tMalware Count :', val_table['ground_truth'].value_counts().iloc[1])
print('\tGoodware Count:', val_table['ground_truth'].value_counts().iloc[0])

# 创建验证集数据集
valid_dataset = ExeDataset(list(val_table.index), valid_data_path, list(val_table.ground_truth), first_n_byte)

# 初始化验证集数据加载器
_, valid_loader = init_loader(valid_dataset, batch_size)

# 加载模型
model_path = 'malconv_model_CBP.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MalConv(input_length=first_n_byte, window_size=window_size)
model.load_state_dict(torch.load(model_path))  # 加载已保存的模型
model = model.to(device)

# 在验证集上进行评估
evaluate_model(model, valid_loader, device)
