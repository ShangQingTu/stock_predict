import pandas as pd
import matplotlib.pyplot as plt
import datetime
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from TrainSet import TrainSet
from Model import RNN
import os


def generate_df_affect_by_n_days(series, n, index=False):
    # 过一个序列来生成一个矩阵（用于处理时序的数据）
    # 就是把当天的前n天作为参数，当天的数据作为label
    # 可以不看
    if len(series) <= n:
        raise Exception("The Length of series is %d, while affect by (n=%d)." % (len(series), n))
    df = pd.DataFrame()
    for i in range(n):
        df['c%d' % i] = series.tolist()[i:-(n - i)]
    df['y'] = series.tolist()[n:]
    if index:
        df.index = series.index[n:]
    return df


def readData(column='Adj Close', n=30, all_too=True, index=False, train_end=-300):
    # 从csv读入数据,可以不看
    df = pd.read_csv("韦尔股份603501.ss.csv", index_col=0)
    df.index = list(map(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"), df.index))
    df_column = df[column].copy()
    df_column_train, df_column_test = df_column[:train_end], df_column[train_end - n:]
    df_generate_from_df_column_train = generate_df_affect_by_n_days(df_column_train, n, index=index)
    if all_too:
        return df_generate_from_df_column_train, df_column, df.index.tolist()
    return df_generate_from_df_column_train


def count_dates(predict_days, start_day):
    # 目前仅仅支持月份不变的计算
    # count_dates只是一个生成日期的函数,可以不看
    start_date = start_day.split("-")
    year = int(start_date[0])
    month = int(start_date[1])
    day = int(start_date[2])
    predict_dates = []
    for i in range(predict_days):
        date = "{}-{}-{}".format(year, month, day + i)
        predict_dates.append(date)
    return predict_dates


def count_accuracy(real, generate_data_test):
    # 计算模型在测试集上的准确率
    sumReal = np.sum(real)
    lossTest_raw = real - np.array(generate_data_test)
    lossTest = np.maximum(lossTest_raw, -lossTest_raw)#取误差的绝对值
    sumLoss = np.sum(lossTest)
    return 1 - (sumLoss / sumReal)


if __name__ == '__main__':
    # 主函数入口,运行即开始训练模型,训练一个模型大概要1分钟
    # n表示，模型假设"今天的股票价格与过去n天的股价相关"
    # 这个n可以大胆的修改,没人知道假设多少天是合理的
    n = 20
    # train_end是train集与test集的分界线,表示倒数-train_end个原始数据划分为测试集,剩余是训练集
    # 取60是因为总的数据有650条,测试集一般占10%
    train_end = -60
    # picdic是保存图片的目录,
    # 比如test_days5_n30其中test_days5表示train_end=-5的情况,即test集只有5天,n30表示n=30
    picdic = "./test_days" + str(-train_end) + "_n" + str(n) + "/"
    if not os.path.exists(picdic):
        # 如果不存在就创建文件夹
        os.mkdir(picdic)
    # 要预测未来多少天就修改这个predict_days，它是天数
    predict_days = 7
    # 从"2020-1-1"开始预测
    predict_start_day = "2020-1-1"
    # LR是学习率(learning rate),是模型每一步移动的大小,决定模型是否能收敛到稳定的状态
    LR = 0.0001
    # EPOCH是训练的次数
    EPOCH = 50
    # 数据集建立
    df, df_all, df_index = readData('Adj Close', n=n, train_end=train_end)
    df_all = np.array(df_all.tolist())
    plt.plot(df_index, df_all, label='real-data')
    df_numpy = np.array(df)
    # 平均值和标准差
    df_numpy_mean = np.mean(df_numpy)
    df_numpy_std = np.std(df_numpy)
    # 归一化为正态分布,归一化是为了便于训练模型
    df_numpy = (df_numpy - df_numpy_mean) / df_numpy_std
    df_tensor = torch.Tensor(df_numpy)
    # 加载数据为tensor格式
    trainset = TrainSet(df_tensor)
    trainloader = DataLoader(trainset, batch_size=10, shuffle=True)

    # rnn = torch.load('rnn.pkl')#这个是调用现成训练好的模型才会用的,可以忽视

    # 模型的训练
    rnn = RNN(n)
    # 优化器选择Adam优化器,这个是pytorch写好的优化器，直接调用,lr是学习率
    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
    # 损失函数定义为均方损失,是pytorch写好的，直接调用
    loss_func = nn.MSELoss()
    # 把数据正向输入EPOCH次来训练
    for step in range(EPOCH):
        for tx, ty in trainloader:
            output = rnn(torch.unsqueeze(tx, dim=0))  # 一次正向通过模型的输出
            loss = loss_func(torch.squeeze(output), ty)  # 计算输出与真实值(label)的差异
            optimizer.zero_grad()  # 因为optimizer会积累梯度，所以每个step清零一次
            loss.backward()  # 反向传播,计算梯度
            optimizer.step()  # 优化器进行一步优化
        # 打印每一步的loss,loss越小,模型和训练集的拟合程度越高,但是过高的话会导致过拟合"overfitting"
        print('Epoch [{}/{}],  Loss: {:.4f}'
              .format(step + 1, EPOCH, loss.item()))
    # 保存模型
    torch.save(rnn, 'rnn.pkl')
    # 下面进行模型验证,使用的是test集,上面训练用到的是train集
    generate_data_train = []
    generate_data_test = []
    generate_data_predict = []
    # train_end是train集与test集的分界线
    test_index = len(df_all) + train_end
    # df_all是原始数据,归一化为正态分布df_all_normal
    df_all_normal = (df_all - df_numpy_mean) / df_numpy_std
    df_all_normal_tensor = torch.Tensor(df_all_normal)

    for i in range(n, len(df_all) + predict_days):
        x = df_all_normal_tensor[i - n:i]
        x = torch.unsqueeze(torch.unsqueeze(x, dim=0), dim=0)
        y = rnn(x)
        if i < test_index:
            # generate_data_train是模型在前test_index天预测的股价,看模型在训练集上的表现
            generate_data_train.append(torch.squeeze(y).detach().numpy() * df_numpy_std + df_numpy_mean)
        elif i < len(df_all):
            # generate_data_train是模型在倒数-train_end天预测的股价,看模型在测试集上的表现
            generate_data_test.append(torch.squeeze(y).detach().numpy() * df_numpy_std + df_numpy_mean)
        else:
            # generate_data_predict是预测未来的股价
            now_data = torch.squeeze(y).detach().numpy() * df_numpy_std + df_numpy_mean
            generate_data_predict.append(now_data)
            temp = []
            temp.append((now_data - df_numpy_mean) / df_numpy_std)
            # 预测一天,就要把df_all_normal更新一天
            df_all_normal = np.insert(df_all_normal, 0, values=np.array(temp), axis=0)
            df_all_normal_tensor = torch.Tensor(df_all_normal)

    # 下面是画图过程,画了3个图
    plt.plot(df_index[n:train_end], generate_data_train, label='generate_train')
    plt.plot(df_index[train_end:], generate_data_test, label='generate_test')
    plt.legend()
    # 这个图是模型在训练集和测试集的表现
    plt.savefig(picdic + "trainAndTest.png")
    plt.show()
    plt.cla()
    plt.plot(df_index[train_end:], df_all[train_end:], label='real-data')
    plt.plot(df_index[train_end:], generate_data_test[:], label='generate_test')
    plt.legend()
    # 计算在test集上的差异
    accuracy = count_accuracy(df_all[train_end:], generate_data_test)
    print('Accuracy of the network on the test set: {} %'.format(100 * accuracy))
    # 这个图是在训练集上的表现,其实只是放大了来看
    plt.savefig(picdic + "test.png")
    plt.show()
    # 从2020-1-1开始预测predict_days天，这个count_dates只是一个生成日期的函数
    predict_dates = count_dates(predict_days, predict_start_day)
    with open(picdic + "股价.txt", "w") as res_w:
        # 保存预测的股价
        res_w.write('Accuracy of the network on the test set: {} %'.format(100 * accuracy))
        res_w.write("\n")
        res_w.write("generate_data_predict")
        for stockPrice in generate_data_predict:
            res_w.write("\n")
            res_w.write(str(stockPrice))
        res_w.write("\n")
        res_w.write("predict_dates")
        for date in predict_dates:
            res_w.write("\n")
            res_w.write(str(date))
    print("generate_data_predict", generate_data_predict)
    print("predict_dates", predict_dates)
    plt.cla()
    plt.plot(predict_dates, generate_data_predict, label='generate_predict')
    plt.legend()
    # 这个图是模型对未来的预测
    plt.savefig(picdic + "predict.png")
    plt.show()
