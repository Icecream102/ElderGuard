import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils import data
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

# 解决colab引用路径问题
import sys
sys.path.append('/content/drive/MyDrive/ElderGuard/Human-Falling-Detect-Tracks-master/Actionsrecognition')
sys.path.append('/content/drive/MyDrive/ElderGuard/Human-Falling-Detect-Tracks-master')
from Actionsrecognition.Models import *

# from Visualizer import plot_graphs, plot_confusion_metrix

data_name = 'test_data_2.pkl'
test_path = '/content/drive/MyDrive/ElderGuard/data/input_data/' + data_name
# data_name = 'test_data_2.pkl'
# test_path = 'D:/learn/ciscn/data/skeleton/input_data/' + data_name


# test_result.csv记录训练集上的实验数据
def to_result(model_i, los, acc, pre, rec, f1, name):
    result_path = '/content/drive/MyDrive/ElderGuard/data/test_result.csv'
    result_data = {'model': model_i, 'loss': los, 'accuracy': acc, 'precision': pre, 'recall': rec, 'f1-score': f1,
                   'data_name': name}
    r = pd.DataFrame(result_data, index=[0])
    r.to_csv(result_path, mode='a', index=False, header=False)


def dataset_loader(path, batch_size):
    features, labels = [], []

    with open(path, 'rb') as f:
        fts, lbs = pickle.load(f)
        features.append(fts)
        labels.append(lbs)

    del fts, lbs

    features = np.concatenate(features, axis=0)  # 多个数组的拼接
    labels = np.concatenate(labels, axis=0)

    data_set = data.TensorDataset(torch.tensor(features, dtype=torch.float32).permute(0, 3, 1, 2),  # permute:进行转置
                                  torch.tensor(labels, dtype=torch.float32))
    data_loader = data.DataLoader(data_set, batch_size, shuffle=True)

    return data_loader


def set_training(model, mode=True):
    for p in model.parameters():
        p.requires_grad = mode
    model.train(mode)
    return model


def accuracy_batch(y_pred, y_true):
    return (y_pred.argmax(1) == y_true.argmax(1)).mean()


# class_names = ['Standing', 'Walking', 'Sitting', 'Lying Down',
#                'Stand up', 'Sit down', 'Fall Down', 'violence']  # 添加了violence
class_names = ['Standing', 'Walking', 'Sitting', 'Lying Down',
               'Stand up', 'Sit down', 'Fall Down']
num_class = len(class_names)
device = 'cuda'
model_path = 'saved/model0'
batch_size = 32
if __name__ == '__main__':
    # MODEL.
    graph_args = {'strategy': 'spatial'}
    model = TwoStreamSpatialTemporalGraph(graph_args, num_class).to(device)

    model.load_state_dict(torch.load(os.path.join(model_path, 'tsstg-model.pth')))

    # 交叉熵损失函数
    losser = torch.nn.BCELoss()

    # EVALUATION.
    model = set_training(model, False)
    eval_loader = dataset_loader(test_path, batch_size)

    print('Evaluation.')
    run_loss = 0.0
    run_accu = 0.0
    y_preds = []
    y_trues = []
    true_label_list = []
    pred_label_list = []
    with tqdm(eval_loader, desc='eval') as iterator:
        for pts, lbs in iterator:
            mot = pts[:, :2, 1:, :] - pts[:, :2, :-1, :]
            mot = mot.to(device)
            pts = pts.to(device)
            lbs = lbs.to(device)

            out = model((pts, mot))
            loss = losser(out, lbs)

            run_loss += loss.item()
            accu = accuracy_batch(out.detach().cpu().numpy(),
                                  lbs.detach().cpu().numpy())
            run_accu += accu

            y_preds.extend(out.argmax(1).detach().cpu().numpy())
            y_trues.extend(lbs.argmax(1).cpu().numpy())

            iterator.set_postfix_str(' loss: {:.4f}, accu: {:.4f}'.format(
                loss.item(), accu))
            iterator.update()

    run_loss = run_loss / len(iterator)
    run_accu = run_accu / len(iterator)
    print('Eval Loss: {:.4f}, Accu: {:.4f}'.format(run_loss, run_accu))

    # 精确度、召回率、F1分数
    precision, recall, f1_score = precision_recall_fscore_support(y_trues, y_preds, average='macro')[:-1]
    print("precision: ", precision)
    print("recall: ", recall)
    print("f1_score: ", f1_score)
    to_result(model_path, run_loss, run_accu, precision, recall, f1_score, data_name)  # 保存实验记录

    report = classification_report(y_trues, y_preds, digits=4)
    print(report)

# plot_confusion_metrix(y_true=y_trues, y_pred=y_preds,
#
