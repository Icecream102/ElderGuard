import os
import pickle
import pandas as pd
from tqdm import tqdm
from torch.utils import data
from torch.optim.adadelta import Adadelta
from torch.optim.lr_scheduler import ReduceLROnPlateau  # 动态调整学习率
from torch.utils.tensorboard import SummaryWriter

# 解决colab引用路径问题
import sys

sys.path.append('/content/drive/MyDrive/ElderGuard/Human-Falling-Detect-Tracks-master/Actionsrecognition')
sys.path.append('/content/drive/MyDrive/ElderGuard/Human-Falling-Detect-Tracks-master')
from Actionsrecognition.Models import *
from Visualizer import plot_graphs, plot_confusion_metrix

# 需要修改/调整
model_idx = 'model1'  # 训练的第几个模型
save_folder = '/content/drive/MyDrive/ElderGuard/Human-Falling-Detect-Tracks-master/Actionsrecognition/saved/' + model_idx  # 训练结果保存的文件夹
tensorboard_log = '/content/drive/MyDrive/ElderGuard/logs' + '/' + model_idx  # tensorboard可视化
checkpoint_path = '/content/drive/MyDrive/ElderGuard/Human-Falling-Detect-Tracks-master/Actionsrecognition/saved/checkpoint/model1_e29.pth'  # 上次保存的模型参数的文件,如果需要断点续训,这个文件应该为上一次训练最后保存参数的文件
resume = True  # 是否接着上次训练
start_epoch = 0  # 默认从头开始训练
device = 'cuda'  # 使用GPU
epochs = 50
batch_size = 128

class_names = ['Standing', 'Walking', 'Sitting', 'Lying Down',
               'Stand up', 'Sit down', 'Fall Down', 'violence']  # 添加了violence
num_class = len(class_names)

train_path = '/content/drive/MyDrive/ElderGuard/data/input_data/train_data_2.pkl'
val_path = '/content/drive/MyDrive/ElderGuard/data/input_data/valid_data_2.pkl'
test_path = '/content/drive/MyDrive/ElderGuard/data/input_data/test_data_2.pkl'


# result.csv记录实验数据
def to_result(model_i, epoch_i, t_a, v_a, t_l, v_l):
    result_path = '/content/drive/MyDrive/ElderGuard/data/model_result.csv'
    result_data = {'model': model_i, 'epoch': epoch_i, 'train_acc': t_a, 'val_acc': v_a, 'train_loss': t_l, 'val_loss': v_l}
    r = pd.DataFrame(result_data, index=[0])
    r.to_csv(result_path, mode='a', index=False, header=False)


# 分别得到训练集、验证集、测试集的DataLoader
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


def accuracy_batch(y_pred, y_true):
    return (y_pred.argmax(1) == y_true.argmax(1)).mean()


def set_training(model, mode=True):
    for p in model.parameters():
        p.requires_grad = mode
    model.train(mode)
    return model


if __name__ == '__main__':
    save_folder = os.path.join(os.path.dirname(__file__), save_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(tensorboard_log):
        os.makedirs(tensorboard_log)

    loss_list = {'train': [], 'valid': []}
    accu_list = {'train': [], 'valid': []}
    lr_list = []  # 记录学习率的变化

    # DATA.
    train_loader = dataset_loader(train_path, batch_size)
    valid_loader = dataset_loader(val_path, batch_size)
    # train_loader, valid_loader = load_dataset(batch_size)
    dataloader = {'train': train_loader, 'valid': valid_loader}

    # MODEL.
    graph_args = {'strategy': 'spatial'}
    model = TwoStreamSpatialTemporalGraph(graph_args, num_class).to(device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = Adadelta(model.parameters())  # 初始学习率默认为1.0
    scheduler = ReduceLROnPlateau(optimizer, 'min')  # 优化学习率

    # 判断是否为断点续训
    if resume:  #
        checkpoint = torch.load(checkpoint_path)  # 加载断点
        model.load_state_dict(checkpoint['net'])  # # 加载模型可学习参数
        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']  # 设置开始的epoch
        loss_list = checkpoint['loss_list']  # 用于绘图
        accu_list = checkpoint['acc_list']  # 用于绘图
        print('finish loaded... start with epoch{}'.format(start_epoch))

    # 交叉熵损失函数
    losser = torch.nn.BCELoss()

    # 添加tensorboard
    writer = SummaryWriter(tensorboard_log)  #

    # TRAINING.
    for e in range(start_epoch + 1, epochs):
        lr_list.append(optimizer.param_groups[0]["lr"])  # 获取当前优化器中使用的学习率
        print('Epoch {}/{}'.format(e, epochs - 1))
        for phase in ['train', 'valid']:
            if phase == 'train':
                model = set_training(model, True)
            else:
                model = set_training(model, False)

            run_loss = 0.0
            run_accu = 0.0
            with tqdm(dataloader[phase], desc=phase) as iterator:  # tqdm实现进度条
                for pts, lbs in iterator:
                    # Create motion input by distance of points (x, y) of the same node
                    # in two frames.
                    mot = pts[:, :2, 1:, :] - pts[:, :2, :-1, :]

                    mot = mot.to(device)
                    pts = pts.to(device)
                    lbs = lbs.to(device)

                    # Forward.
                    out = model((pts, mot))

                    # RuntimeError: Found dtype Float but expected Long
                    loss = losser(out, lbs)

                    if phase == 'train':
                        # Backward. 优化器优化模型
                        model.zero_grad()  # 将上一轮的梯度清零
                        loss.backward()  # 反向计算出各参数的梯度
                        optimizer.step()  # 更新网络中的参数

                    run_loss += loss.item()
                    accu = accuracy_batch(out.detach().cpu().numpy(),  # detach():阻断反传
                                          lbs.detach().cpu().numpy())
                    run_accu += accu

                    iterator.set_postfix_str(' loss: {:.4f}, accu: {:.4f}'.format(
                        loss.item(), accu))
                    iterator.update()
                    # break

            if phase == 'valid':  #
                valid_loss = run_loss / len(iterator)
                scheduler.step(valid_loss)  # 需要在优化器参数更新之后再动态调整学习率

            loss_list[phase].append(run_loss / len(iterator))
            accu_list[phase].append(run_accu / len(iterator))
            # break

        # 每个epoch后将相应训练结果添加到tensorboard和result.csv
        writer.add_scalar("train_loss", loss_list['train'][-1], e)
        writer.add_scalar("valid_loss", loss_list['valid'][-1], e)
        writer.add_scalar("train_accuracy", accu_list['train'][-1], e)
        writer.add_scalar("valid_accuracy", accu_list['valid'][-1], e)
        to_result(model_idx, e, accu_list['train'][-1], accu_list['valid'][-1], loss_list['train'][-1],
                  loss_list['valid'][-1])
        print('Summary epoch:\n - Train loss: {:.4f}, accu: {:.4f}\n - Valid loss:'
              ' {:.4f}, accu: {:.4f}'.format(loss_list['train'][-1], accu_list['train'][-1],
                                             loss_list['valid'][-1], accu_list['valid'][-1]))
        print('lr_list: ', lr_list)

        # SAVE.
        # torch.save(model.state_dict(), os.path.join(save_folder, 'tsstg-model.pth')) # 我改成每个epoch的都分开保存，原来的会覆盖上一轮的模型
        # 在coloab上我一边训练一边删掉结果不怎么好的模型
        name = 'tsstg-model' + str(e) + '.pth'
        torch.save(model.state_dict(), os.path.join(save_folder, name))
        # 每5个epoch结束后保存一次模型参数，实现断点续训
        if e % 5 == 4:
            checkpoint = {
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': e,
                'loss_list': loss_list,
                'acc_list': accu_list
            }
            # checkpoint_path = '/saved/checkpoint/model1_e1.pth'
            if not os.path.isdir(
                    "/content/drive/MyDrive/ElderGuard/Human-Falling-Detect-Tracks-master/Actionsrecognition/saved/checkpoint"):
                os.mkdir(
                    "/content/drive/MyDrive/ElderGuard/Human-Falling-Detect-Tracks-master/Actionsrecognition/saved/checkpoint")
            c_path = '/content/drive/MyDrive/ElderGuard/Human-Falling-Detect-Tracks-master/Actionsrecognition/saved/checkpoint/' + model_idx + '_e' + str(
                e) + '.pth'
            torch.save(checkpoint, c_path)

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

        # break
    writer.close()  #
    del train_loader, valid_loader