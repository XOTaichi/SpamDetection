import csv
import random

# 增加 CSV 字段大小限制
csv.field_size_limit(10**7)  # 设置为 10MB，或者更高

# 定义文件路径
legit_file = r'raw_dataset\apollo\legit.csv'
spam_file = r'raw_dataset\apollo\phishing.csv'
output_file = r'dataset\body\apollo\combined_data.csv'

# 读取 CSV 文件并提取数据
def read_csv(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 检查字段是否存在并且非空
            if 'body' not in row or len(row['body']) == 0:
                continue  # 如果 body 字段缺失或为空，跳过该行
            # 处理大字段，避免超出字段大小限制
            if len(row['body']) > 10000:  # 假设超过 10,000 字符的字段会被认为过大
                row['body'] = row['body'][:10000]  # 截断字段为 10,000 字符
            data.append(row)
    return data

# 从两份文件中读取数据
legit_data = read_csv(legit_file)
spam_data = read_csv(spam_file)

# 确保数据量足够，否则抛出错误
if len(legit_data) < 1000 or len(spam_data) < 1000:
    raise ValueError("One of the datasets contains fewer than 1000 entries.")

# 随机选择 1000 条数据
legit_sample = random.sample(legit_data, 1000)
spam_sample = random.sample(spam_data, 1000)

# 创建新的数据集合
combined_data = []

# 给每条数据添加一个 id 并标记其类别
for i, row in enumerate(legit_sample):
    combined_data.append({'id': i + 1, 'body': row['body'], 'label': 'ham'})

for i, row in enumerate(spam_sample):
    combined_data.append({'id': len(legit_sample) + i + 1, 'body': row['body'], 'label': 'spam'})

# 写入新的 CSV 文件
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    fieldnames = ['id', 'body', 'label']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(combined_data)

print(f"New dataset created at: {output_file}")
