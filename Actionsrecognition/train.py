import os
import time
import torch
import pickle
import numpy as np
import torch.nn.functional as F
from shutil import copyfile
from tqdm import tqdm
from torch.utils import data
from torch.optim.adadelta import Adadelta
from sklearn.model_selection import train_test_split

from Actionsrecognition.Models import *
from Visualizer import plot_graphs, plot_confusion_metrix

"""
train
    coffee100%+home20%
valid
    home80%
test
    home100%
"""

"""
TwoStreamSpatialTemporalGraph
"""


save_folder = 'Actionsrecognition/saved/TSSTG(pts+mot)-01(cf+hm-hm)'


"""
hyper parameters
"""
device = 'cuda'
epochs = 10
batch_size = 32

# DATA FILES.
# Should be in format of
#  inputs: (N_samples, time_steps, graph_node, channels),
#  labels: (N_samples, num_class)
#   and do some of normalizations on it. Default data create from:
#       Data.create_dataset_(1-3).py
# where
#   time_steps: Number of frame input sequence, Default: 30
#   graph_node: Number of node in skeleton, Default: 14
#   channels: Inputs data (x, y and scores), Default: 3
#   num_class: Number of pose class to train, Default: 7

data_files = ['Data/tmp/Home_new-set(labelXscrw).pkl',
              'Data/tmp/Home_new-set(labelXscrw).pkl']
# file = 'Data/tmp/Home_new-set(labelXscrw).pkl'
class_names = ['Standing', 'Walking', 'Sitting', 'Lying Down',
               'Stand up', 'Sit down', 'Fall Down']
num_class = len(class_names)


def load_dataset(data_files, batch_size, split_size=0):
    """
    Load data files into torch DataLoader with/without spliting train-test.
    """
    features, labels = [], []
    for fil in data_files:
        with open(fil, 'rb') as f:  # Open .pkl file which saved features and labels
            fts, lbs = pickle.load(f)
            features.append(fts)
            labels.append(lbs)
        del fts, lbs

    features = np.concatenate(features, axis=0)  # features.shape = (203, 30, 14, 3)
    labels = np.concatenate(labels, axis=0)  # labels.shape = (203, 7)

    if split_size > 0:
        # total 203, train 162, valid 41
        x_train, x_valid, y_train, y_valid = train_test_split(features, labels, test_size=split_size, random_state=9)
        # print("There are {} training samples".format(y_train.shape[0]))
        # print("There are {} validing samples".format(y_valid.shape[0]))
        train_set = data.TensorDataset(torch.tensor(x_train, dtype=torch.float32).permute(0, 3, 1, 2), torch.tensor(y_train, dtype=torch.float32))
        valid_set = data.TensorDataset(torch.tensor(x_valid, dtype=torch.float32).permute(0, 3, 1, 2), torch.tensor(y_valid, dtype=torch.float32))
        # 封装数据 train_set、valid_set
        train_loader = data.DataLoader(train_set, batch_size, shuffle=True)
        valid_loader = data.DataLoader(valid_set, batch_size)
    else:
        train_set = data.TensorDataset(torch.tensor(features, dtype=torch.float32).permute(0, 3, 1, 2), torch.tensor(labels, dtype=torch.float32))
        train_loader = data.DataLoader(train_set, batch_size, shuffle=True)
        valid_loader = None

    return train_loader, valid_loader


def accuracy_batch(y_pred, y_true):

    return (y_pred.argmax(1) == y_true.argmax(1)).mean()


def set_training(model, mode=True):
    """
    training mode: requires_grad = True
    """
    for p in model.parameters():
        p.requires_grad = mode
    model.train(mode)  # 将网络模型设为训练模式，并且需要梯度

    return model


if __name__ == '__main__':
    # """
    # create save folder
    # """
    # # save_folder = os.path.join(os.path.dirname(__file__), save_folder)
    # # if not os.path.exists(save_folder):
    # #     os.makedirs(save_folder)

    """
    load data
    """
    train_loader, _ = load_dataset(data_files[0:1], batch_size)  # coffee room所有
    # for i, data in enumerate(train_loader, start=1):
    #     x_data, y_label = data
    #     print("batch:{0} x_data:{1} y_label:{2}".format(i, x_data, y_label))  # 7个batch
    valid_loader, train_loader_ = load_dataset(data_files[1:2], batch_size, 0.2)  # home
    train_loader = data.DataLoader(data.ConcatDataset([train_loader.dataset, train_loader_.dataset]), batch_size, shuffle=True)

    dataloader = {'train': train_loader, 'valid': valid_loader}

    del train_loader_

    """
    initialize network
    """
    graph_args = {'strategy': 'spatial'}  # uniform / distance / spatial
    model = TwoStreamSpatialTemporalGraph(graph_args, num_class).to(device)
    # print(list(model.parameters()))

    """
    loss and optimizer
    """
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = Adadelta(model.parameters())
    losser = torch.nn.BCELoss()

    """
    train network
    """
    loss_list = {'train': [], 'valid': []}
    accu_list = {'train': [], 'valid': []}

    for e in range(epochs):
        print('Epoch {}/{}'.format(e+1, epochs))
        for phase in ['train', 'valid']:
            if phase == 'train':
                model = set_training(model, True)
            else:
                model = set_training(model, False)

            run_loss = 0.0
            run_accu = 0.0

            with tqdm(dataloader[phase], desc=phase) as iterator:
                for pts, lbs in iterator:
                    mot = pts[:, :2, 1:, :] - pts[:, :2, :-1, :]   # Create motion input by distance of points (x, y) of the same node in two frames.
                    mot = mot.to(device)  # motions
                    pts = pts.to(device)  # points
                    lbs = lbs.to(device)  # labels
                    # Forward.
                    out = model((pts, mot))
                    loss = losser(out, lbs)
                    # Backward.
                    if phase == 'train':
                        model.zero_grad()
                        loss.backward()
                        # gradiant descent
                        optimizer.step()
                    # loss and accuracy
                    run_loss += loss.item()
                    accu = accuracy_batch(out.detach().cpu().numpy(), lbs.detach().cpu().numpy())
                    run_accu += accu
                    # bar description
                    iterator.set_postfix_str(' loss: {:.4f}, accu: {:.4f}'.format(loss.item(), accu))
                    iterator.update()
                    # break

            loss_list[phase].append(run_loss / len(iterator))
            accu_list[phase].append(run_accu / len(iterator))
            # break: train or valid, get train_loss_list or valid_loss_list

        print('Summary epoch:\n - Train loss: {:.4f}, accu: {:.4f}\n - Valid loss:'
              ' {:.4f}, accu: {:.4f}'.format(loss_list['train'][-1], accu_list['train'][-1],
                                             loss_list['valid'][-1], accu_list['valid'][-1]))

        # save params of model for each epoch
        # torch.save(model.state_dict(), os.path.join(save_folder, 'tsstg-model.pth'))
        torch.save(model.state_dict(), os.path.join(save_folder, 'tsstg-model-{}.pth').format(e+1))

        # visualize the loss and accuracy
        plot_graphs(list(loss_list.values()), list(loss_list.keys()),
                    'Last Train: {:.2f}, Valid: {:.2f}'.format(
                        loss_list['train'][-1], loss_list['valid'][-1]
                    ), 'Loss', xlim=[0, epochs],
                    save=os.path.join(save_folder, 'loss_graph.png'))
        plot_graphs(list(accu_list.values()), list(accu_list.keys()),
                    'Last Train: {:.2f}, Valid: {:.2f}'.format(
                        accu_list['train'][-1], accu_list['valid'][-1]
                    ), 'Accu', xlim=[0, epochs],
                    save=os.path.join(save_folder, 'accu_graph.png'))

        # break: epoch

    del train_loader, valid_loader

    """
    evaluation
    """
    model.load_state_dict(torch.load(os.path.join(save_folder, 'tsstg-model-10.pth')))
    model = set_training(model, False)
    data_file = data_files[1]
    eval_loader, _ = load_dataset([data_file], 32)
    print('Evaluation.')
    run_loss = 0.0
    run_accu = 0.0
    y_preds = []
    y_trues = []
    with tqdm(eval_loader, desc='eval') as iterator:
        for pts, lbs in iterator:
            mot = pts[:, :2, 1:, :] - pts[:, :2, :-1, :]
            mot = mot.to(device)
            pts = pts.to(device)
            lbs = lbs.to(device)
            # forward
            out = model((pts, mot))
            loss = losser(out, lbs)
            # loss and accuracy
            run_loss += loss.item()
            accu = accuracy_batch(out.detach().cpu().numpy(), lbs.detach().cpu().numpy())
            run_accu += accu
            # y_pred and y_true
            y_preds.extend(out.argmax(1).detach().cpu().numpy())  # 阻断反向传播，从GPU转移到CPU，将Tensor转为Numpy
            y_trues.extend(lbs.argmax(1).cpu().numpy())
            # bar description
            iterator.set_postfix_str(' loss: {:.4f}, accu: {:.4f}'.format(loss.item(), accu))
            iterator.update()
            # break

    run_loss = run_loss / len(iterator)
    run_accu = run_accu / len(iterator)

    # plot_confusion_metrix(y_trues, y_preds, class_names, 'Eval on: {}\nLoss: {:.4f}, Accu{:.4f}'.format(os.path.basename(data_file), run_loss, run_accu), 'true', save=os.path.join(save_folder, '{}-confusion_matrix.png'.format(
    #     os.path.basename(data_file).split('.')[0])))

    print('Eval Loss: {:.4f}, Accu: {:.4f}'.format(run_loss, run_accu))
