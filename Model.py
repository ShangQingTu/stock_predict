import torch.nn as nn
class RNN(nn.Module):
    def __init__(self, input_size):
        #这一句是继承
        super(RNN, self).__init__()
        #定义rnn层为LSTM,设置输入大小,隐含层大小等参数
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        #定义输出层为线性层
        self.out = nn.Sequential(
            nn.Linear(64, 1)
        )
        #定义dropout层,这一层的作用是随机舍弃10%学习到的特征,可以比较有效地减轻过拟合的发生
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        #正向传播
        #每一个step,进来的数据会经过rnn层，变成r_out
        r_out, (h_n, h_c) = self.rnn(x, None)  # None 表示 hidden state 会用全0的 state
        # r_out经过dropout层,变成d_out
        d_out = self.dropout(r_out)
        # d_out经过线性层,变成out输出
        out = self.out(d_out)

        return out
