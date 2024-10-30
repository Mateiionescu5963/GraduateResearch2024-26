import os
import pandas as pd

def merge_csv_files(batch_prefixes, batch_range, src_folder, dst_folder):
    """
    将多个CSV文件中的train和test分别合并，并保存到目标文件夹中。
    
    参数：
    batch_prefixes: CSV文件的前缀，例如 ['b_batch_', 'm_batch_']。
    batch_range: batch 文件的编号范围，例如 range(1, 21)。
    src_folder: 源文件夹的根路径。
    dst_folder: 目标文件夹的根路径。
    """
    for i in batch_range:
        # 定义文件名
        train_files = [os.path.join(src_folder, f'{prefix}{i}_train.csv') for prefix in batch_prefixes]
        test_files = [os.path.join(src_folder, f'{prefix}{i}_test.csv') for prefix in batch_prefixes]

        # 读取并合并train文件
        merged_train_df = merge_csv(train_files)
        # 保存合并后的train文件
        merged_train_path = os.path.join(dst_folder, f'batch_{i}_train.csv')
        merged_train_df.to_csv(merged_train_path, index=False)

        # 读取并合并test文件
        merged_test_df = merge_csv(test_files)
        # 保存合并后的test文件
        merged_test_path = os.path.join(dst_folder, f'batch_{i}_test.csv')
        merged_test_df.to_csv(merged_test_path, index=False)

def merge_csv(files):
    """
    读取并合并多个CSV文件。
    
    参数：
    files: 要合并的CSV文件列表。
    
    返回：
    合并后的DataFrame。
    """
    dataframes = []
    for file in files:
        if os.path.exists(file):
            df = pd.read_csv(file)
            dataframes.append(df)
    if dataframes:
        return pd.concat(dataframes, ignore_index=True)
    else:
        return pd.DataFrame()  # 如果没有文件，返回空的DataFrame

# 设置参数
batch_prefixes = ['b_batch_', 'm_batch_']  # CSV文件的前缀
batch_range = range(1, 11)  # batch 文件的编号范围，例如 1 到 20
src_folder = 'data/'  # 源文件夹的根路径
dst_folder = 'data/'  # 目标文件夹的根路径

# 创建目标文件夹（如果不存在）
os.makedirs(dst_folder, exist_ok=True)

# 调用函数
merge_csv_files(batch_prefixes, batch_range, src_folder, dst_folder)
