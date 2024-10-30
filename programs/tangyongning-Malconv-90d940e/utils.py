import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def save_model(model, path):
    torch.save(model.state_dict(), path)

def calculate_result(y_pred, y_true):
    y_pred = torch.sigmoid(y_pred)  # Apply sigmoid to convert logits to probabilities
    y_pred = (y_pred > 0.5).float()  # Binarize predictions at threshold 0.5
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    return accuracy, precision, recall, f1

'''
使用 torch.argmax(y_pred, dim=1) 可能不适合计算二分类的结果，因为这通常用于多分类任务。
对于二分类任务，更常见的做法是将预测值通过阈值,如0.5,进行二值化处理。
如果使用 BCEWithLogitsLoss 作为损失函数，通常模型的输出是未经过 sigmoid 激活函数的原始 logits,
因此在计算结果时需要对输出进行 sigmoid 变换并应用阈值。
'''