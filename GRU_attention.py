import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GRU, Concatenate, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras.layers.merge import Multiply

from loss import *


class GRUWithLocalAttention(Model):
    def __init__(self, center_size=64,output_size=1, hidden_size=128, l2_regularization=0.001, learning_rate=0.01,
                 attention_type='global', input_shape=(None, None)):
        super(GRUWithLocalAttention, self).__init__()
        self.center_size = center_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.l2_regularization = l2_regularization
        self.learning_rate = learning_rate
        self.attention_type = attention_type
        self.build(input_shape=input_shape)  # 动态确定输入形状

    def build(self, input_shape):
        self.input1 = Input(shape=input_shape)
        self.gru_output = self._gru_layer(self.input1)
        self.attention = self._attention_layer(self.gru_output)
        self.fc_output = self._dense_layer(self.attention)
        self.output1 = self._output_layer(self.fc_output)

        super(GRUWithLocalAttention, self).__init__(inputs=self.input1, outputs=self.output1)

    def _gru_layer(self, x):
        return GRU(units=self.center_size-self.output_size, return_sequences=True)(x)

    def _attention_layer(self, x):
        if self.attention_type == 'global':
            attention = self._global_attention(x)
        else:
            attention = self._local_attention(x)
        return attention

    def _global_attention(self, x):
        scores = Dense(units=x.shape[-1], activation='tanh')(x)
        scores = Dense(units=x.shape[-1], activation='softmax')(scores)
        return Multiply()([scores, x])

    def _local_attention(self, x):
        output_dim = x.shape[-1]
        output_len = x.shape[1] - 5 + 1
        conv = Conv1D(filters=output_dim, kernel_size=5, padding='valid')
        scores = conv(x)
        scores = Dense(units=x.shape[-1], activation='softmax')(scores)
        return Multiply()([scores, x])

    def _dense_layer(self, x):
        return Dense(units=self.hidden_size, activation='relu', kernel_regularizer=l2(self.l2_regularization))(x)

    def _output_layer(self, x):
        return Dense(units=1, activation='linear', kernel_regularizer=l2(self.l2_regularization))(x)

    def compile(self, loss='mean_squared_error', metrics=['mae']):
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        super(GRUWithLocalAttention, self).compile(optimizer=optimizer, loss=loss, metrics=metrics)

# # 使用示例
# model = GRUWithLocalAttention(center_size=64, hidden_size=128, l2_regularization=0.001, learning_rate=0.01,
#                               attention_type='local')
# model.compile(loss='mean_squared_error', metrics=['mae'])
# model.summary()
