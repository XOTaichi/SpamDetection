import pandas as pd

def extract_and_merge_data(input_csv, output_csv, max_count=700):
    # 读取原始 CSV 数据，尝试使用 'ISO-8859-1' 编码
    try:
        df = pd.read_csv(input_csv, encoding='ISO-8859-1')
    except UnicodeDecodeError as e:
        print(f"Error reading the file: {e}")
        return

    
    # 过滤 ham 和 spam 数据
    ham_data = df[df['v1'] == 'ham']
    spam_data = df[df['v1'] == 'spam']
    
    # 获取每类数据的实际条数
    ham_count = min(len(ham_data), max_count)
    spam_count = min(len(spam_data), max_count)
    
    # 抽取最多 1000 条数据
    ham_data = ham_data.sample(n=ham_count, random_state=42)
    spam_data = spam_data.sample(n=spam_count, random_state=42)
    
    # 合并数据
    merged_data = pd.concat([ham_data, spam_data], ignore_index=True)
    
    # 保存合并后的数据到新的 CSV 文件
    merged_data.to_csv(output_csv, index=False)
    print(f"Extracted and merged {ham_count} ham and {spam_count} spam samples. Saved to {output_csv}")

# 输入和输出文件路径
input_csv = 'raw_dataset\\SMS\\spam.csv'
output_csv = 'dataset\\merged_data.csv'

# 调用函数
extract_and_merge_data(input_csv, output_csv)
