import os
import random
import shutil
import pandas as pd

# 原始文件夹路径
source_folder = '../../../data/DikeData/files/'  # 替换为实际的文件夹路径
# 目标文件夹路径
output_folder = '../../../data/DikeData/processed_data/'  # 替换为你想存储结果的文件夹路径

# CSV文件保存路径
csv_output_folder = '../../../data/DikeData/'  # 替换为你想保存CSV文件的路径

# 获取文件夹中所有文件的文件名
all_files = os.listdir(source_folder)

# 随机抽取10份，每份1000个不重复文件
num_files_per_batch = 11923
num_batches = 1
train_ratio = 0.8  # 80% 作为训练集，20% 作为测试集

# 检查是否有足够的文件
if len(all_files) < num_files_per_batch * num_batches:
    raise ValueError("文件夹中的文件数量不足，无法完成抽样")

# 创建输出目录
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 创建CSV文件保存目录
if not os.path.exists(csv_output_folder):
    os.makedirs(csv_output_folder)

for i in range(1, num_batches + 1):
    # 从剩下的文件中随机抽取1000个不重复的文件
    sampled_files = random.sample(all_files, num_files_per_batch)
    
    # 从总文件列表中删除已抽取的文件，避免重复
    all_files = [file for file in all_files if file not in sampled_files]
    
    # 为每批文件创建一个独立的文件夹
    batch_folder = os.path.join(output_folder, f'b_batch_{i}')
    os.makedirs(batch_folder, exist_ok=True)
    
    # 创建train和test子文件夹
    train_folder = os.path.join(batch_folder, 'train')
    test_folder = os.path.join(batch_folder, 'test')
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    
    # 将文件按比例分配到训练集和测试集
    num_train_files = int(num_files_per_batch * train_ratio)
    train_files = sampled_files[:num_train_files]
    test_files = sampled_files[num_train_files:]
    
    # 创建CSV文件内容
    train_csv_data = []
    test_csv_data = []
    
    # 处理训练集文件
    for file in train_files:
        file_source_path = os.path.join(source_folder, file)
        file_target_path = os.path.join(train_folder, file)
        
        # 将文件从原始文件夹复制到train文件夹
        shutil.copy(file_source_path, file_target_path)
        
        # 将文件名和label写入train CSV数据列表
        train_csv_data.append([file, 1])
    
    # 处理测试集文件
    for file in test_files:
        file_source_path = os.path.join(source_folder, file)
        file_target_path = os.path.join(test_folder, file)
        
        # 将文件从原始文件夹复制到test文件夹
        shutil.copy(file_source_path, file_target_path)
        
        # 将文件名和label写入test CSV数据列表
        test_csv_data.append([file, 1])
    
    # 创建CSV文件路径
    train_csv_file_path = os.path.join(csv_output_folder, f'b_batch_{i}_train.csv')
    test_csv_file_path = os.path.join(csv_output_folder, f'b_batch_{i}_test.csv')
    
    # 将训练集文件名和label写入train CSV文件
    train_df = pd.DataFrame(train_csv_data, columns=['filename', 'label'])
    train_df.to_csv(train_csv_file_path, index=False)
    
    # 将测试集文件名和label写入test CSV文件
    test_df = pd.DataFrame(test_csv_data, columns=['filename', 'label'])
    test_df.to_csv(test_csv_file_path, index=False)

    print(f'Batch {i} created with {len(train_files)} train files and {len(test_files)} test files. CSV saved to {csv_output_folder}')
