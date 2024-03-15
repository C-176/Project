import tensorflow as tf

weight_list = []


def amplitude_shifting_invariance_with_softmax(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    # 计算预测结果和真实结果的距离
    distances = tf.abs(y_true - y_pred)
    length = tf.cast(len(y_true), dtype=tf.float32)

    # 使用 softmax 函数对距离进行归一化，使所有时刻的距离之和为 1
    softmax_distances = tf.nn.softmax(distances, axis=0)

    # 计算损失
    loss = tf.reduce_mean(length * tf.reduce_sum(tf.abs(1 / length - softmax_distances)))
    # loss = tf.reduce_mean(length * tf.reduce_sum(tf.abs(1 / length - softmax_distances) * weight_list))

    return loss


def invarias_with_fourier_coefficients(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.complex64)
    y_pred = tf.cast(y_pred, dtype=tf.complex64)
    # 对时间序列进行傅里叶变换
    fft_true = tf.signal.fft(y_true)
    fft_pred = tf.signal.fft(y_pred)

    # 获取预测结果和真实结果的主成分
    pc_true = tf.abs(fft_true) / tf.reduce_sum(tf.abs(fft_true))
    pc_pred = tf.abs(fft_pred) / tf.reduce_sum(tf.abs(fft_pred))

    # 使用范数对比两个序列的主成分差异作为损失函数
    loss = tf.reduce_sum(tf.norm(pc_true - pc_pred, ord='euclidean'))
    # loss = tf.reduce_sum(tf.norm(pc_true - pc_pred, ord='euclidean') * weight_list)

    return loss


def corr(x, y, axis=0, win=5):
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.float32)
    x = x[:-win]
    if axis == 1:
        y = y[win:]
    else:
        y = y[:-win]

    mean_x = tf.reduce_mean(x)
    mean_y = tf.reduce_mean(y)
    autocov = tf.reduce_sum((x - mean_x) * (y - mean_y)) / tf.cast(tf.shape(y)[0], dtype=tf.float32)
    autocorr = autocov / tf.sqrt(tf.math.reduce_variance(x) * tf.math.reduce_variance(y))
    return autocorr


def invarias_with_auto_correlation(y_true, y_pred):
    # 计算真实序列的自相关系数
    autocorr = corr(y_true, y_true, axis=1)

    # 计算预测结果和真实序列的相关系数
    xycorr = corr(y_true, y_pred, axis=0)

    # 计算损失
    loss = tf.norm(autocorr - xycorr)

    return loss


def tilde_q_loss(y_true, y_pred):
    global weight_list
    num = len(y_true)
    weight_list = tf.cast(range(num), dtype=tf.float32)
    mse = tf.losses.MeanSquaredError()
    loss = (
            0.1 * amplitude_shifting_invariance_with_softmax(y_true, y_pred)
            + 0.9 * invarias_with_fourier_coefficients(y_true, y_pred)
            + 0.01 * invarias_with_auto_correlation(y_true, y_pred)
            + 0.01 * mse.call(y_true, y_pred)
    )
    return loss


if __name__ == '__main__':
    seq = tf.constant(range(10))
    seq2 = tf.constant(range(1, 11))
    print(invarias_with_fourier_coefficients(seq, seq2))
