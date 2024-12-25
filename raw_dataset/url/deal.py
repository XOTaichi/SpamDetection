import csv
import random

# 定义文件路径
input_file = r'raw_dataset\url\urldata.csv'
train_file = r'dataset\url\train.csv'
test_file = r'dataset\url\test.csv'

# 读取 CSV 文件并按标签分类
good_data = []
bad_data = []

# 读取文件并分类数据
with open(input_file, 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)  # 使用 DictReader 方便按列名访问数据
    for row in reader:
        label = row['label']
        if label == 'good':
            good_data.append(row)
        elif label == 'bad':
            bad_data.append(row)

# 保证每个类别有相同数量的数据
min_length = min(len(good_data), len(bad_data))
good_data = good_data[:min_length]
bad_data = bad_data[:min_length]

# 合并两类数据
combined_data = good_data + bad_data

# 打乱数据
random.shuffle(combined_data)

# 划分训练集和测试集（80% 训练集，20% 测试集）
split_index = int(0.95 * len(combined_data))
train_data = combined_data[:split_index]
test_data = combined_data[split_index:]

# 获取字段名（即列名）
fieldnames = reader.fieldnames  # 从文件头部获取列名

# 写入 train.csv 文件
with open(train_file, 'w', newline='', encoding='utf-8') as train_csv:
    writer = csv.DictWriter(train_csv, fieldnames=fieldnames)
    writer.writeheader()  # 写入表头
    writer.writerows(train_data)  # 写入数据

# 写入 test.csv 文件
with open(test_file, 'w', newline='', encoding='utf-8') as test_csv:
    writer = csv.DictWriter(test_csv, fieldnames=fieldnames)
    writer.writeheader()  # 写入表头
    writer.writerows(test_data)  # 写入数据

print(f"Data split into {train_file} and {test_file} successfully!")
