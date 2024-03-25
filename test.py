import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable


# 定义一个包含Attention机制的LSTM层
class LSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True):
        super(LSTMWithAttention, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=batch_first)
        self.attention = nn.Linear(hidden_size, hidden_size)
        # self.attention = nn.Linear(hidden_size, 1)  # 用于计算变量维度的注意力分数

    def forward(self, input_seq):
        # LSTM的输出
        lstm_output, (h_n, c_n) = self.lstm(input_seq)  # (batch_size, seq_len, hidden_size)

        attention_scores = self.attention(lstm_output)  # (batch_size, seq_len, hidden_size)
        attention_scores = torch.bmm(attention_scores, attention_scores.permute(0, 2, 1))
        attention_scores = F.softmax(attention_scores, dim=-1)
        weighted_output = torch.bmm(attention_scores, lstm_output)  # (batch_size, seq_len, hidden_size)

        # 返回加权输出
        return weighted_output

        # # 应用变量维度的注意力机制
        # attention_scores = self.attention(lstm_output)  # (batch_size, seq_len, 1)
        # attention_scores = torch.sigmoid(attention_scores)  # 归一化得到注意力分数
        # weighted_output = attention_scores * lstm_output  # 应用注意力权重
        #
        # # 取最后一个时间步的输出作为最终输出
        # final_output = weighted_output[:, -1, :]  # (batch_size, hidden_size)
        # return final_output

    # 定义点积Attention函数
    def att_dot_seq_len(self, x):
        # 计算点积
        e = torch.bmm(x, x.permute(0, 2, 1))
        # 归一化得到注意力分数
        attention = F.softmax(e, dim=-1)
        return attention


# 生成合成数据的函数
def generate_data(batch_size, seq_len, input_size, output_size):
    # 生成随机输入数据
    X_train = torch.randn(batch_size, seq_len, input_size)
    X_test = torch.randn(batch_size, seq_len, input_size)

    # 生成随机输出数据（这里我们简单地使用输入数据作为输出）
    y_train = X_train
    y_test = X_test

    return X_train, y_train, X_test, y_test


# 可视化预测结果的函数
def visualize_results(y_test, y_pred, seq_len):
    # 绘制真实值和预测值
    for i in range(seq_len):
        plt.plot(y_test[:, i], label=f'True_{i}')
        plt.plot(y_pred[:, i], label=f'Pred_{i}')
    plt.legend()
    plt.show()


# 主函数
def main1(X_train=None, y_train=None, X_test=None, y_test=None):
    # 超参数设置
    batch_size = 10
    seq_len = 1
    input_size = 14
    output_size = 1
    hidden_size = 14
    num_layers = 1
    learning_rate = 0.001
    num_epochs = 1000

    # 创建模型实例
    model = LSTMWithAttention(input_size, hidden_size, num_layers)
    X_train, y_train, X_test, y_test = torch.from_numpy(X_train).double(), torch.from_numpy(y_train).double(), torch.from_numpy(X_test).double(), torch.from_numpy(y_test).double()


    X_train, y_train, X_test, y_test = generate_data(batch_size, seq_len, input_size, output_size)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # 训练模型
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

    # 评估模型
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        predictions = model(X_test)
        test_loss = criterion(predictions, y_test)

    print(f'Test Loss: {test_loss.item()}')

    # 可视化预测结果
    visualize_results(y_test[:, :, 0], predictions[:, :, 0], seq_len)


if __name__ == "__main__":
    main1()
