import pandas as pd

splits = {'train': 'train.jsonl', 'test': 'test.jsonl'}
train_data_path = "hf://datasets/SetFit/enron_spam/" + splits["train"]
test_data_path = "hf://datasets/SetFit/enron_spam/" + splits["test"]

df = pd.read_json(train_data_path, lines=True)

df.to_json(r"dataset\body\enorn\train_data.json", orient="records", lines=True)

df = pd.read_json(test_data_path, lines=True)

df.to_json(r"dataset\body\enorn\test_data.json", orient="records", lines=True)

print("数据已成功保存到本地文件。")
