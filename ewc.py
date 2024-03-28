import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback

from tensorflow.keras.losses import MeanSquaredError  # 正确的导入方式
from tensorflow.keras import backend as K

from Plot import Plot


#
# def compute_fisher_information_matrix(model, data, labels):
#     # 获取模型的输出
#     with tf.GradientTape() as tape:
#         predictions = model(data)
#         loss = MeanSquaredError()(labels, predictions)  # 使用正确的损失函数类
#     gradients = tape.gradient(loss, model.trainable_weights)
#
#     # 计算FIM的元素
#     num_params = sum(w.shape[0] * w.shape[1] for w in model.trainable_weights)
#     fisher_info = np.zeros((num_params, num_params))
#
#     for i, (dw_i, w_i) in enumerate(zip(gradients, model.trainable_weights)):
#         w_i_flat = w_i.flatten()
#         for j, (dw_j, w_j) in enumerate(zip(gradients, model.trainable_weights)):
#             w_j_flat = w_j.flatten()
#             # 计算FIM的(i, j)元素
#             fisher_info[i * num_params + j, i * num_params + j] = np.mean(dw_i * dw_j)
#
#     return fisher_info
# 假设我们有一个函数来计算FIM，这里我们使用一个简化的版本
def compute_fisher_information_matrix(model, data, labels):
    # 获取模型的权重
    weights = [w.numpy() for w in model.trainable_weights]

    # 初始化FIM
    num_params = len(weights)
    fisher_info = np.zeros((num_params, num_params))

    # 计算损失函数相对于每个参数的梯度
    with tf.GradientTape() as tape:
        predictions = model(data)
        loss = tf.keras.losses.MeanSquaredError(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_weights)

    # 计算FIM的元素
    for i, (dw_i, w_i) in enumerate(zip(gradients, weights)):
        for j, (dw_j, w_j) in enumerate(zip(gradients, weights)):
            # 计算FIM的(i, j)元素
            fisher_info[i, j] = np.mean(dw_i * dw_j, axis=0)

    return fisher_info


# 定义EWC正则化回调
class EWC(Callback):
    def __init__(self, fisher_blockdiag, lambda_ewc):
        super(EWC, self).__init__()
        self.fisher_blockdiag = fisher_blockdiag
        self.lambda_ewc = lambda_ewc
        self.ewc_reg = self.create_ewc_regularizer()

    def create_ewc_regularizer(self):
        def regularizer(weights):
            return self.lambda_ewc * np.sum(self.fisher_blockdiag * np.square(weights))

        return regularizer

    def on_train_begin(self, logs=None):
        # 在训练开始前，添加EWC正则化项
        self.model.compiled_loss = self.model.compiled_loss + self.ewc_reg(self.model.trainable_weights)

    def on_epoch_end(self, epoch, logs=None):
        # 更新FIM
        new_fisher_info = compute_fisher_information_matrix(self.model, self.model.history['x'],
                                                            self.model.history['y'])
        self.fisher_blockdiag = new_fisher_info


# 定义网络结构
input_shape = (None, 1)  # 假设输入是变长的一维数组
inputs = Input(shape=input_shape)
gru = GRU(64, return_sequences=False)(inputs)
outputs = Dense(1, activation='linear')(gru)
model = Model(inputs=inputs, outputs=outputs)

# 编译模型
model.compile(optimizer='adam', loss='mean_squared_error')

# 假设我们有训练数据
X_train = np.random.rand(1000, 1)
y_train = np.random.rand(1000, 1)

# 训练模型
model.fit(X_train, y_train, epochs=10, batch_size=32)

# 计算初始FIM
fisher_info_first_month = compute_fisher_information_matrix(model, X_train, y_train)

# 设置EWC正则化超参数
lambda_ewc = 1.0

# 创建EWC回调
ewc_callback = EWC(fisher_blockdiag=fisher_info_first_month, lambda_ewc=lambda_ewc)

# 假设我们有新的数据用于增量学习
X_new = np.random.rand(1000, 1)
y_new = np.random.rand(1000, 1)

# 使用EWC进行增量学习
model.fit(X_new, y_new, epochs=10, batch_size=32, callbacks=[ewc_callback])

# 评估模型性能
loss = model.evaluate(X_new, y_new, verbose=0)
r = model.predict(X_new)
Plot.paint_double('xx', show=True, data_dict={
    'true': y_new, 'pred': r
})
