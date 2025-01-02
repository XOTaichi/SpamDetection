from cnn import CNN
import torch
import random
import numpy as np
import sklearn.metrics as metrics
import pandas as pd
import matplotlib.pyplot as plt
import os
from lstm import LSTM
import time

PATH = './runs/'
IGNORE = {'vocab_size', 'num_classes', 'labels', 'print_per_batch', 'model_type'}

class TextTrain:
    def __init__(self, config, train_X, train_Y, test_X, test_Y, is_url=False):
        self.config = config
        self.train_X = torch.tensor(train_X, dtype=torch.long)
        self.train_Y = torch.tensor(train_Y, dtype=torch.long)
        self.test_X = torch.tensor(test_X, dtype=torch.long)
        self.test_Y = torch.tensor(test_Y, dtype=torch.long)
        self.setup_seed(1234)
        self.__build_model()

        path = os.path.join(PATH, config['model_type'])
        tmp = ''.join([f'{key}_{config[key]}' for key in config if key not in IGNORE])
        if is_url:
            tmp = 'url_' + tmp
        path = os.path.join(path, tmp)
        os.makedirs(path, exist_ok=True)
        self.path = path

    @staticmethod
    def setup_seed(seed):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    @staticmethod
    def evaluate(criterion, outputs, targets):
        loss = criterion(outputs, targets)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == targets).sum().item()
        accuracy = correct / targets.size(0)
        return loss.item(), accuracy


    def print_report_table(self, preds, targets, labels, avg_inference_time=None):
        p, r, f1, s = metrics.precision_recall_fscore_support(targets, preds, labels=range(len(labels)))
        tot_p = np.average(p, weights=s)
        tot_r = np.average(r, weights=s)
        tot_f1 = np.average(f1, weights=s)
        tot_s = np.sum(s)
        res1 = pd.DataFrame({
            'Label': labels,
            'Precision': p,
            'Recall': r,
            'F1': f1,
            'Support': s
        })
        res2 = pd.DataFrame({
            'Label': ['总体'],
            'Precision': [tot_p],
            'Recall': [tot_r],
            'F1': [tot_f1],
            'Support': [tot_s]
        })
        res2.index = [len(res1)]
        res = pd.concat([res1, res2], ignore_index=True)
        print('\n Report Table:')
        print(res)

        Report_path = os.path.join(self.path, 'report_table.txt')
        with open(Report_path, 'w') as f:
            f.write(res.to_string())
            if avg_inference_time is not None:
                f.write(f'\n\n Average Inference Time: {avg_inference_time:.4f} ms/sample')


    def print_confusion_matrix(self, preds, targets, labels):
        cm = metrics.confusion_matrix(targets, preds)
        df = pd.DataFrame(cm, columns=labels, index=labels)
        print('\n Confusion Matrix:')
        print(df)
        Confusion_path = os.path.join(self.path, 'confusion_matrix.txt')
        with open(Confusion_path, 'w') as f:
            f.write(df.to_string())


    def plot_roc_curve(self, outputs, targets):
        plt.clf()
        # 获取正类的预测概率
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().detach().numpy()
        y_true = targets.cpu().numpy()
        fpr, tpr, _ = metrics.roc_curve(y_true, probs)
        roc_auc = metrics.auc(fpr, tpr)

        Figure1_path = os.path.join(self.path, 'roc_curve.png')
        plt.figure(figsize=(10, 10))
        plt.plot(fpr, tpr, color='darkorange',
                 lw=2, label='ROC curve (area = %0.2f)' % roc_auc)  # 假正率为横坐标，真正率为纵坐标做曲线
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")

        plt.savefig(Figure1_path)
        plt.close()
    

    def plot_loss_curve(self, losses):
        Figure2_path = os.path.join(self.path, 'loss_curve.png')
        plt.clf()
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(losses)), losses, '-b', label='Training Loss')
        plt.title('Training Loss over Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(Figure2_path)
        plt.close()
    
    def __build_model(self):
        """根据配置选择模型"""
        torch.cuda.empty_cache()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model_type = self.config.get('model_type', 'cnn').lower()
        if model_type == 'cnn':
            self.model = CNN(
                embedding_dim=self.config['embedding_dim'], 
                vocab_size=self.config['vocab_size'],
                num_filters=self.config['num_filters'], 
                kernel_size=self.config['kernel_size'], 
                hidden_dim=self.config['hidden_dim'], 
                dropout_keep_prob=self.config['dropout_keep_prob'],
                num_classes=self.config['num_classes']
            ).to(self.device)
        elif model_type == 'lstm':
            self.model = LSTM(
                embedding_dim=self.config['embedding_dim'], 
                vocab_size=self.config['vocab_size'], 
                hidden_dim=self.config['hidden_dim'], 
                dropout_keep_prob=self.config.get('dropout', 0.5),
                num_classes=self.config['num_classes'], 
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config['learning_rate'], momentum=0.9)

    def train(self):
        """通用的模型训练方法"""
        max_accuracy = 0
        losses = []
        model_type = self.config.get('model_type', 'model')  # 获取模型类型，默认为 'model'
        model_save_path = os.path.join(self.path, f'best_{model_type}.pth')  # 根据模型类型动态生成路径

        for i in range(self.config['num_iterations']):
            selected_index = random.sample(range(len(self.train_X)), k=self.config['batch_size'])
            batch_X = self.train_X[selected_index].to(self.device)
            batch_Y = self.train_Y[selected_index].to(self.device)

            outputs = self.model(batch_X)
            self.optimizer.zero_grad()
            loss = self.criterion(outputs, batch_Y)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

            step = i + 1
            if step % self.config['print_per_batch'] == 0 or step == 1:
                loss_val, accuracy = self.evaluate(self.criterion, outputs, batch_Y)
                print(f'step:{step} loss:{loss_val:.4f} accuracy:{accuracy:.4f}')

                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    torch.save(self.model.state_dict(), model_save_path)
        self.plot_loss_curve(losses)

    def test(self):
        """通用的模型测试方法"""
        model_type = self.config.get('model_type', 'model')  # 获取模型类型，默认为 'model'
        model_load_path = os.path.join(self.path, f'best_{model_type}.pth')  # 根据模型类型动态生成路径

        self.model.load_state_dict(torch.load(model_load_path, map_location=self.device))
        self.model.eval()

        batch_times = []
        with torch.no_grad():
            batch_size = 32
            num_batches = len(self.test_X) // batch_size
            all_outputs = []

            # 测试时间
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = (i + 1) * batch_size
                batch_X = self.test_X[start_idx:end_idx].to(self.device)
                
                # 记录每个batch的推理时间
                start_time = time.time()
                outputs = self.model(batch_X)
                end_time = time.time()
                batch_times.append(1000 * (end_time - start_time))
                
                all_outputs.append(outputs)
                
            if len(self.test_X) % batch_size != 0:
                batch_X = self.test_X[num_batches * batch_size:].to(self.device)
                
                start_time = time.time()
                outputs = self.model(batch_X)
                end_time = time.time()
                batch_times.append(1000 * (end_time - start_time))
                
                all_outputs.append(outputs)

            outputs = torch.cat(all_outputs, dim=0)
            test_Y = self.test_Y.to(self.device)

            total_time = sum(batch_times)
            avg_time = total_time / len(self.test_X)

            print(f"\n推理性能统计:")
            print(f"测试集总样本数: {len(self.test_X)}")
            print(f"总推理时间: {total_time:.4f} ms")
            print(f"平均推理时间: {avg_time:.4f} ms/样本")

            # 计算准确率和损失
            loss, accuracy = self.evaluate(self.criterion, outputs, test_Y)
            print(f'测试损失: {loss:.4f} 准确率: {accuracy:.4f}')

            # 输出分类报告
            preds = torch.argmax(outputs, dim=1)
            self.print_report_table(preds.cpu().numpy(), test_Y.cpu().numpy(), self.config['labels'], avg_time)

            # 混淆矩阵
            self.print_confusion_matrix(preds.cpu().numpy(), test_Y.cpu().numpy(), self.config['labels'])

            # ROC曲线(二分类情况)
            if len(self.config['labels']) == 2:
                self.plot_roc_curve(outputs, test_Y)
                