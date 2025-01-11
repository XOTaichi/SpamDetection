# 代码说明
* 复现了https://github.com/jpmorganchase/llm-email-spam-detection的工作，并对main.py和training.py进行了修改以满足我们工作的需求
* 复现的工作本身自带6种machine learning 方法和3种LM方法，我们主要选用了NB、KNN和RoBERTa
* 如果想复现这一部分，只需按照原repo配置环境，替换我们的main.py和training.py即可
* 四个process为前缀的脚本是用于处理我们的数据集，转化成复现工作所需的数据集的脚本，运行python process_body.py file/to/input file/to/output 即可使用
