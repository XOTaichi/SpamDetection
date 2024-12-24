import os
import tarfile
import urllib.request

# 下载数据集
def download_data(url, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    file_name = url.split("/")[-1]
    target_path = os.path.join(target_folder, file_name)
    if not os.path.exists(target_path):
        urllib.request.urlretrieve(url, target_path)
        print(f"Downloaded {file_name}")
    return target_path

# 解压数据集
def extract_data(file_path, target_folder):
    with tarfile.open(file_path, "r:bz2") as tar:
        tar.extractall(target_folder)
        print(f"Extracted {os.path.basename(file_path)} to {target_folder}")

# 下载并解压数据集
url_list = [
    "https://spamassassin.apache.org/old/publiccorpus/20021010_easy_ham.tar.bz2",
    "https://spamassassin.apache.org/old/publiccorpus/20021010_hard_ham.tar.bz2",
    "https://spamassassin.apache.org/old/publiccorpus/20021010_spam.tar.bz2"
]

target_folder = "raw_dataset"
for url in url_list:
    file_path = download_data(url, target_folder)
    extract_data(file_path, target_folder)
