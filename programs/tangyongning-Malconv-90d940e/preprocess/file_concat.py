import os
import shutil

def merge_folders(batch_prefixes, batch_range, src_folder, dst_folder):
    """
    将多个 batch 文件夹的 test 和 train 合并到目标文件夹中。
    
    参数：
    batch_prefixes: batch 文件夹的前缀，例如 ['b_batch_', 'm_batch_']。
    batch_range: batch 文件夹的编号范围，例如 range(1, 11)。
    src_folder: 源文件夹的根路径。
    dst_folder: 目标文件夹的根路径。
    """
    for i in batch_range:
        # 定义源和目标文件夹路径
        b_batch = os.path.join(src_folder, f'{batch_prefixes[0]}{i}')
        m_batch = os.path.join(src_folder, f'{batch_prefixes[1]}{i}')
        target_batch = os.path.join(dst_folder, f'batch_{i}')

        # 创建目标文件夹
        os.makedirs(target_batch, exist_ok=True)

        # 定义test和train的合并路径
        target_test = os.path.join(target_batch, 'test')
        target_train = os.path.join(target_batch, 'train')

        # 创建test和train文件夹
        os.makedirs(target_test, exist_ok=True)
        os.makedirs(target_train, exist_ok=True)

        # 合并test文件夹
        merge_subfolders([b_batch, m_batch], 'test', target_test)

        # 合并train文件夹
        merge_subfolders([b_batch, m_batch], 'train', target_train)

def merge_subfolders(src_folders, subfolder_name, dst_folder):
    """
    将多个源文件夹中的指定子文件夹合并到目标文件夹中。
    
    参数：
    src_folders: 源文件夹的列表。
    subfolder_name: 要合并的子文件夹名，例如 'test' 或 'train'。
    dst_folder: 合并后的目标文件夹。
    """
    for src_folder in src_folders:
        src_subfolder = os.path.join(src_folder, subfolder_name)
        if os.path.exists(src_subfolder):
            for item in os.listdir(src_subfolder):
                s = os.path.join(src_subfolder, item)
                d = os.path.join(dst_folder, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)

# 设置参数
batch_prefixes = ['b_batch_', 'm_batch_']  # 前缀
batch_range = range(1, 11)  # batch 编号范围
src_folder = 'data/'  # 源文件夹的根路径
dst_folder = 'data/'  # 目标文件夹的根路径

# 调用函数
merge_folders(batch_prefixes, batch_range, src_folder, dst_folder)
