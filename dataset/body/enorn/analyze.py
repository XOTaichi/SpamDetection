import json
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

# 读取数据
with open(r'dataset\body\enorn\train_data.jsonl','r') as trainfile:
    train_data = [json.loads(line) for line in trainfile.readlines()]
with open(r'dataset\body\enorn\test_data.jsonl','r') as testfile:
    test_data = [json.loads(line) for line in testfile.readlines()]

# 统计标签分布
train_labels = [item['label_text'] for item in train_data]
test_labels = [item['label_text'] for item in test_data]

train_label_counts = Counter(train_labels)
test_label_counts = Counter(test_labels)

print("Train Label Counts:", train_label_counts)
print("Test Label Counts:", test_label_counts)

total_train = len(train_data)
total_test = len(test_data)
train_ratio = total_train / (total_train + total_test)
test_ratio = total_test / (total_train + total_test)

print(f"Train Data Ratio: {train_ratio:.2%}")
print(f"Test Data Ratio: {test_ratio:.2%}")

def plot_pie_chart(label_counts, title="Label Distribution"):
    labels = label_counts.keys()
    sizes = label_counts.values()
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title(title)
    plt.show()

plot_pie_chart(train_label_counts, title="Train Label Distribution")
plot_pie_chart(test_label_counts, title="Test Label Distribution")

# 生成词云
from wordcloud import STOPWORDS

train_text = " ".join([item['message'] for item in train_data])
test_text = " ".join([item['message'] for item in test_data])

stop_words = set(STOPWORDS)  # 使用词云库自带的停用词集合
stop_words.update(stopwords.words('english'))  # 如果需要英文停用词也可以加进去
custom_stopwords = ["enron",'said']
stop_words.update(custom_stopwords)

def generate_wordcloud(text):
    wordcloud = WordCloud(stopwords=stop_words, background_color="white", width=800, height=400).generate(text)
    word_frequencies = wordcloud.words_

    top_10_words = sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)[:10]

    # 可视化词云
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Word Cloud with Top 10 Words")
    print(top_10_words)
    # # 在词云图的右侧显示Top 10词汇
    # x_offset = 1.05  # 偏移量，将文本显示在词云图旁边
    # y_offset = 0.9  # 调整文本显示的起始位置
    # for i, (word, freq) in enumerate(top_10_words):
    #     plt.text(x_offset, y_offset - i * 0.05, f"{word}: {freq:.2f}", fontsize=12, color="black", ha="left", va="top")

    plt.show()

generate_wordcloud(train_text)
generate_wordcloud(test_text)

# 分析文本长度
train_text_lengths = [len(item['message']) for item in train_data]
test_text_lengths = [len(item['message']) for item in test_data]

plt.figure(figsize=(10, 5))
plt.hist(train_text_lengths, bins=50, alpha=0.5, label='Train Data')
plt.hist(test_text_lengths, bins=50, alpha=0.5, label='Test Data')
plt.xlabel('Message Length')
plt.ylabel('Frequency')
plt.legend()
plt.show()
