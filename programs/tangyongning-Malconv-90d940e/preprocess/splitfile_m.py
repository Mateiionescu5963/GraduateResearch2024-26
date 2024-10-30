import os
import shutil
import csv
import random
from sklearn.model_selection import train_test_split

def set_read_only(file_path):
    os.chmod(file_path, 0o444)  # 将文件权限修改为只读

def split_and_label_files_to_csv(source_directory, train_directory, test_directory, test_size=0.2, max_files=1200):
    # 获取所有文件名
    filenames = os.listdir(source_directory)
    
    # 如果文件数量大于 max_files，则随机选择 max_files 个文件
    if len(filenames) > max_files:
        filenames = random.sample(filenames, max_files)
    
    # 分割文件名为训练集和测试集
    train_filenames, test_filenames = train_test_split(filenames, test_size=test_size)
    
    # 创建训练集和测试集文件夹
    os.makedirs(train_directory, exist_ok=True)
    os.makedirs(test_directory, exist_ok=True)
    
    # 移动训练集文件并修改权限
    for filename in train_filenames:
        src_path = os.path.join(source_directory, filename)
        dest_path = os.path.join(train_directory, filename)
        shutil.move(src_path, dest_path)
        set_read_only(dest_path)
    
    # 移动测试集文件并修改权限
    for filename in test_filenames:
        src_path = os.path.join(source_directory, filename)
        dest_path = os.path.join(test_directory, filename)
        shutil.move(src_path, dest_path)
        set_read_only(dest_path)
    
    # 创建CSV文件并写入文件名和标签
    def write_csv(file_path, filenames):
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['filename', 'label'])
            for filename in filenames:
                writer.writerow([filename, 1])
    
    train_csv_path = os.path.join(train_directory, 'm_train_labels.csv')
    test_csv_path = os.path.join(test_directory, 'm_test_labels.csv')
    
    write_csv(train_csv_path, train_filenames)
    write_csv(test_csv_path, test_filenames)
    
    return train_csv_path, test_csv_path

source_directory = '/home/dgy/Desktop/DikeDataset-main/files/malware'  
train_directory = '/home/dgy/Desktop/Malconv/total_data_m_t'    
test_directory = '/home/dgy/Desktop/Malconv/total_data_m_v'      

train_csv, test_csv = split_and_label_files_to_csv(source_directory, train_directory, test_directory)
print(f"Train CSV saved at: {train_csv}")
print(f"Test CSV saved at: {test_csv}")
