import os
import pandas as pd


def xlsx_to_csv(input_file, output_file):
    # 读取XLSX文件
    df = pd.read_excel(input_file)

    # 将数据保存为CSV文件
    df.to_csv(output_file, index=False)


# 获取当前目录下的所有文件
current_directory = os.getcwd()
files = os.listdir(current_directory)

# 遍历文件列表
for file in files:
    # 检查文件扩展名是否为XLSX
    if file.endswith('.xlsx'):
        # 构建输入和输出文件路径
        input_file = os.path.join(current_directory, file)
        output_file = os.path.splitext(input_file)[0] + '.csv'

        # 转换XLSX文件为CSV
        xlsx_to_csv(input_file, output_file)