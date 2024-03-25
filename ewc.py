from tensorflow.python.keras.models import Model
import tensorflow.python.keras.backend as K
import numpy as np
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, GRU, Dense
from tensorflow.python.keras.callbacks import Callback
from sklearn.metrics import mean_squared_error


def compute_fisher_information_matrix(model, data, labels, num_samples=1000):
    """
    计算给定数据集上的Fisher信息矩阵的近似值。

    参数:
    model -- 已经编译并训练过的Keras模型。
    data -- 输入数据，形状应与模型的输入层相匹配。
    labels -- 真实标签。
    num_samples -- 用于估计FIM的样本数量。

    返回:
    fim -- FIM的近似值。
    """
    # 确保模型已经编译
    # assert model._compiled, 'Model must be compiled before calling this function.'

    # 计算模型输出
    outputs = model.output
    predictions = K.function([model.input], [outputs])
    outputs_value = predictions([data])[0]

    # 计算损失相对于每个参数的梯度
    grads = K.gradients(outputs_value, model.trainable_weights)

    # 初始化Hessian矩阵
    hessian_matrix = K.zeros((len(model.trainable_weights), len(model.trainable_weights)))

    # 对于每个样本，计算Hessian矩阵的估计值
    for i in range(num_samples):
        # 随机扰动标签
        noisy_labels = labels + K.random_normal(K.shape(labels), mean=0., stddev=0.01)

        # 计算扰动后的输出
        noisy_outputs = K.function([model.input, K.variable(noisy_labels)], [outputs])
        noisy_outputs_value = noisy_outputs([data, noisy_labels])[0]

        # 计算Hessian矩阵的梯度
        hessian_vector = K.gradients(noisy_outputs_value, model.trainable_weights)

        # 更新Hessian矩阵的估计值
        hessian_matrix += K.mean(K.outer(hessian_vector, hessian_vector), axis=0)

    # 计算FIM的近似值
    fim = K.batch_flatten(K.eye(len(model.trainable_weights)) + hessian_matrix / (2 * num_samples))

    # 评估FIM
    fim_value = K.function([model.input, K.variable(labels)], [fim])
    fim_values = fim_value([data, labels])[0]

    return fim_values


def ewc_regularizer(fisher_blockdiag, lambda_ewc):
    def regularizer(weights):
        return lambda_ewc * np.sum(fisher_blockdiag * (weights - weights[:, None, None]) ** 2)

    return regularizer


# 定义一个回调类来实现EWC正则化
class EWC(Callback):
    def __init__(self, fisher_blockdiag, lambda_ewc):
        super(EWC, self).__init__()
        self.fisher_blockdiag = fisher_blockdiag
        self.lambda_ewc = lambda_ewc
        self.ewc_reg = ewc_regularizer(self.fisher_blockdiag, self.lambda_ewc)

    def on_train_begin(self, logs=None):
        # 在训练开始前，添加EWC正则化项
        self.model.loss = self.model.loss + self.ewc_reg

    def on_epoch_end(self, epoch, logs=None):
        # 在每个epoch结束时更新FIM
        # 这里需要根据您的数据和模型实现具体的更新方法
        new_fisher_info = compute_fisher_information_matrix(self.model, self.model.trainable_weights,
                                                            self.model.trainable_weights)
        self.fisher_blockdiag = new_fisher_info


# 定义网络结构
input_shape = (None, 1)  # 假设输入是变长的一维数组，例如时间序列数据
inputs = Input(shape=input_shape)
gru = GRU(64, return_sequences=False)(inputs)
outputs = Dense(1, activation='linear')(gru)
model = Model(inputs=inputs, outputs=outputs)

# 编译模型
model.compile(optimizer='adam', loss='mean_squared_error')

# 假设您已经有了第一个月的数据和标签
X_train_first_month = np.random.rand(1000, 1)  # 水质数据
y_train_first_month = np.random.rand(1000, 1)  # 矾量作为输出

# 训练第一个月的数据
model.fit(X_train_first_month, y_train_first_month, epochs=10, batch_size=32)

# 假设您现在有了新的一个月的数据
X_train_new_month = np.random.rand(1000, 1)  # 新的水质数据
y_train_new_month = np.random.rand(1000, 1)  # 新的矾量数据

# 计算第一个月的FIM
fisher_info_first_month = compute_fisher_information_matrix(model, X_train_first_month, y_train_first_month)

# 设置EWC正则化超参数
lambda_ewc = 1.0

# 创建EWC回调
ewc_callback = EWC(fisher_blockdiag=fisher_info_first_month, lambda_ewc=lambda_ewc)

# 使用EWC进行增量学习
model.fit(X_train_new_month, y_train_new_month, epochs=10, batch_size=32, callbacks=[ewc_callback])

# 评估模型性能
loss = model.evaluate(X_train_new_month, y_train_new_month, verbose=0)
print(f'Loss: {loss}')

# 预测新数据
X_test = np.random.rand(100, 1)
y_pred = model.predict(X_test)
print(f'Predictions: {y_pred}')
