import numpy as np
from sklearn import metrics  # 评价指标


class Metrics:

    R2 = 0
    MSE = 0
    RMSE = 0

    def __init__(self):
        pass

    def MAD(self, y_true, y_pred):
        """
        计算每个数据点与目标值的绝对差值的平均值MAD
        """
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs(y_true - y_pred))

    def std(self, data):
        """
        计算标准差
        """
        return np.std(data)

    def cv(self, data):
        """
        计算变异系数 CV (反映序列稳定性的指标，CV越小，序列稳定性越好)
        """
        return np.std(data) / np.mean(data)

    def rmse(self, y_true, y_pred):
        """
        计算RMSE（Root Mean Squared Error）
        :param y_true: 真实值
        :param y_pred: 预测值
        :return: RMSE
        """
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    def absolute_percentage_error(self):
        """
        计算APE（Absolute Percentage Error）
        """
        pass

    def r2(self, y_true, y_pred):
        """
        计算决定系数R2（R-Squared）
        """
        r2 = metrics.r2_score(y_true, y_pred)
        return r2

    def mape(self, y_true, y_pred):
        """
        计算MAPE（Mean Absolute Percentage Error）
        :param y_true: 真实值
        :param y_pred: 预测值
        :return: MAPE
        """
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
