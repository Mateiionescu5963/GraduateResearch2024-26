import pandas as pd
import os

# 定义CSV文件和PE文件夹路径
csv_file = 'data/valid-label.csv'  # CSV文件路径
pe_folder = 'data/valid_optional_headers'  # PE文件所在的文件夹路径
output_csv = 'data/valid-label-headers.csv'  # 输出CSV文件路径

# 读取CSV文件
df = pd.read_csv(csv_file)

# 获取文件夹中的所有文件名（不包括路径）
files_in_folder = set(os.listdir(pe_folder))

# 删除第一列中不在文件夹中的文件名对应的行
df_filtered = df[df.iloc[:, 0].apply(lambda filename: filename in files_in_folder)]

# 将过滤后的DataFrame保存回CSV文件
df_filtered.to_csv(output_csv, index=False)

print(f"处理完成，结果已保存到 {output_csv}")
