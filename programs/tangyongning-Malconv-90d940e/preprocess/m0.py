import os
import pandas as pd

def modify_benign_csv_labels(src_folder, target_label=0, wrong_label=1, benign_prefix='b_batch'):
    """
    只修改benign文件的标签,将错误的label 1 修改为正确的label 0
    
    参数：
    src_folder: CSV文件所在的文件夹路径。
    target_label: 正确的标签值，默认为0。
    wrong_label: 错误的标签值，默认为1。
    benign_prefix: 用于识别benign文件的前缀，默认为 'b_batch'。
    """
    for filename in os.listdir(src_folder):
        if filename.endswith('.csv') and filename.startswith(benign_prefix):
            file_path = os.path.join(src_folder, filename)
            
            # 读取CSV文件
            df = pd.read_csv(file_path)
            
            # 检查是否有label列并修改
            if 'label' in df.columns:
                # 修改错误的label
                df['label'] = df['label'].replace(wrong_label, target_label)
                
                # 将修改后的DataFrame保存回CSV
                df.to_csv(file_path, index=False)
                print(f'Modified {filename}')
            else:
                print(f"No 'label' column found in {filename}")

# 设置参数
src_folder = 'data/'  # 替换为CSV文件所在的路径
correct_label = 0  # 正确的label
incorrect_label = 1  # 误设的错误label
benign_prefix = 'b_batch'  # 用于识别benign文件的前缀

# 执行批量修改
modify_benign_csv_labels(src_folder, target_label=correct_label, wrong_label=incorrect_label, benign_prefix=benign_prefix)
