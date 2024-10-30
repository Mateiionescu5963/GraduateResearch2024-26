import pefile
import os
import concurrent.futures

# 定义包含PE文件的文件夹路径
pe_folder_path = 'data/train/'
optional_header_dir = 'optional_headers'
section_data_dir = 'sections'

# 创建保存二进制数据的文件夹
os.makedirs(optional_header_dir, exist_ok=True)
os.makedirs(section_data_dir, exist_ok=True)

def process_pe_file(file_path):
    filename = os.path.basename(file_path)
    
    try:
        # 解析PE文件
        pe = pefile.PE(file_path)
        
        # 提取 Optional Header 并保存
        optional_header_data = pe.write()[pe.OPTIONAL_HEADER.get_file_offset():pe.OPTIONAL_HEADER.get_file_offset() + pe.OPTIONAL_HEADER.sizeof()]
        with open(os.path.join(optional_header_dir, f'{filename}'), 'wb') as f:
            f.write(optional_header_data)
        
        # 将所有节的二进制数据合并保存为一个文件
        combined_section_data = b''
        for section in pe.sections:
            section_data = section.get_data()
            combined_section_data += section_data
        
        # 将所有节数据写入同一个文件
        with open(os.path.join(section_data_dir, f'{filename}'), 'wb') as f:
            f.write(combined_section_data)
        
        print(f"{filename} 的数据提取完成。")
        
    except pefile.PEFormatError:
        print(f"{filename} 不是有效的 PE 文件，跳过。")
    except Exception as e:
        print(f"处理 {filename} 时出错: {e}")

# 获取文件夹中的所有 PE 文件
pe_files = [os.path.join(pe_folder_path, f) for f in os.listdir(pe_folder_path) if f.endswith(('.exe', '.dll'))]

# 指定线程数为8，创建线程池
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    executor.map(process_pe_file, pe_files)

print("所有文件处理完成。")
