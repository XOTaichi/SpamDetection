import json
import subprocess
import csv
import os
from shutil import rmtree

# 输入 JSONL 文件和输出文件
input_jsonl = "dataset/body/enorn/test_data.jsonl"
output_csv = "result/rule_based_spam_check.csv"
temp_dir = "result/temp_emails"

# 创建临时文件目录
os.makedirs(temp_dir, exist_ok=True)

# 初始化计数器
tp = fp = tn = fn = 0

def check_spam_with_spamassassin(email_body):
    """使用 SpamAssassin 检查邮件 body 是否为垃圾邮件"""
    try:
        # 将邮件内容写入临时文件
        temp_file = os.path.join(temp_dir, "temp_email.txt")
        with open(temp_file, "w") as f:
            f.write(email_body)
        
        # 使用 spamc 检查邮件
        process = subprocess.run(["spamc"], input=email_body, text=True, capture_output=True)
        output = process.stdout

        # 判断是否为垃圾邮件
        return "X-Spam-Status: Yes" in output
    except Exception as e:
        print(f"Error checking spam: {e}")
        return False

# 处理 JSONL 文件
with open(input_jsonl, "r", encoding="utf-8") as infile, \
     open(output_csv, "w", newline="", encoding="utf-8") as outfile:
    # 写入 CSV 表头
    fieldnames = ["message_id", "label_text", "predict"]
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    
    i = 1
    for line in infile:
        print(f"Task {i} done")
        i += 1
        
        # 读取 JSONL 文件的每一行
        row = json.loads(line)
        email_body = row["text"] 
        actual_label = row["label_text"] 
        
        # 检查是否是垃圾邮件
        is_spam = check_spam_with_spamassassin(email_body)
        row["predict"] = "spam" if is_spam else "ham"
        
        # 写入输出 CSV 文件
        filtered_row = {key: row[key] for key in fieldnames}
        
        # 写入输出 CSV 文件
        writer.writerow(filtered_row)
        
        # 更新 TP/FP/TN/FN
        if is_spam and actual_label == 'spam':
            tp += 1
        elif is_spam and actual_label == "ham":
            fp += 1
        elif not is_spam and actual_label == 'ham':
            tn += 1
        elif not is_spam and actual_label == "spam":
            fn += 1

        print(tp,fp,tn,fn)
        print(f"predict label:{is_spam} and actual label:{actual_label}")
# 清理临时文件
rmtree(temp_dir)

# 输出统计结果
print(f"Spam check completed. Results saved to {output_csv}")
print("\n=== Statistics ===")
print(f"True Positives (TP): {tp}")
print(f"False Positives (FP): {fp}")
print(f"True Negatives (TN): {tn}")
print(f"False Negatives (FN): {fn}")

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("\n=== Metrics ===")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1_score:.2f}")
