import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
import nltk
from wordcloud import STOPWORDS

nltk.download('stopwords')

# 读取数据
with open(r'dataset\body\enorn\train_data.jsonl','r') as trainfile:
    train_data = [json.loads(line) for line in trainfile.readlines()]
with open(r'dataset\body\enorn\test_data.jsonl','r') as testfile:
    test_data = [json.loads(line) for line in testfile.readlines()]

# 合并训练集和测试集数据
combined_data = train_data + test_data

# 过滤出label_text为"spam"和"ham"的数据
spam_data = [item['message'] for item in combined_data if item['label_text'] == 'spam']
ham_data = [item['message'] for item in combined_data if item['label_text'] == 'ham']

# 生成合并后的文本
spam_text = " ".join(spam_data)
ham_text = " ".join(ham_data)

# 自定义停用词
stop_words = set(STOPWORDS)
stop_words.update(stopwords.words('english'))  # 加入英文停用词
custom_stopwords = ["enron", 'said','_ _', 'hou ect','ect ect','_ _']
stop_words.update(custom_stopwords)

def generate_wordcloud(text, title="Word Cloud"):
    # 生成词云
    wordcloud = WordCloud(stopwords=stop_words, background_color="white", width=800, height=400).generate(text)
    word_frequencies = wordcloud.words_

    top_10_words = sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)[:10]
    print(top_10_words)
    # 可视化词云
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title)
    plt.show()

# 分别生成 spam 和 ham 的词云
generate_wordcloud(spam_text, title="Spam Word Cloud")
generate_wordcloud(ham_text, title="Ham Word Cloud")
