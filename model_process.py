import logging
import pickle
import random
import shutil
from enum import Enum
from itertools import accumulate
from logging.handlers import TimedRotatingFileHandler

import matplotlib.pyplot as plt
import ruamel
from ruamel import yaml
from scipy.interpolate import make_interp_spline
from sklearn import preprocessing
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.backend import sparse_categorical_crossentropy
from tensorflow.python.keras.callbacks import *
from tensorflow.python.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from tensorflow.python.keras.models import *
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.layers.base import *
from tensorflow.python.keras import losses

from EWC import ewc
from Metrics import Metrics
from Plot import Plot
from loss import *
from utils import *
from callback import *
from public_util import *

import copy
import traceback
import tensorflow as tf
from tensorflow.python.keras.utils.generic_utils import get_custom_objects

import pandas as pd

import os
from datetime import datetime

import pymysql

from sklearn.model_selection import GridSearchCV, KFold
# train_test_split:将数据分为测试集和训练集  GridSearchCV:网格搜索和交叉验证  KFold:K折交叉验证,将训练/测试数据集划分为n个互斥子集，每次用其中一个子集当作验证集，剩下的n-1个作为训练集，进行n次训练和测试，得到n个结果
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics  # 评价指标
import time

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import logging
from scipy.stats import f_oneway
from scipy.stats import spearmanr


def sub_demension_label(*args):
    if not isinstance(args, tuple):
        return Exception(f'传入的参数不是tuple,不可降维')
        # [x,3,1]->[x,1]

    ans = []
    for data in args:
        feature_nums = data.shape[2]
        out = []
        for i in range(len(data)):
            out.append(data[i][-1][-feature_nums:])
        data = np.array(out).reshape(len(out), feature_nums)

        ans.append(data)
    return ans


def add_demension(trainX, time_step=1):
    if not isinstance(trainX, np.ndarray):
        raise Exception(f'trainX不是np.ndarray,没有二维,不可扩展')

    # 将多维数据转换为LSTM需要的三维数据
    len = trainX.shape[0]
    output = []

    for i in range(len - time_step + 1):
        output.append(trainX[i: i + time_step])
    # trainX = np.expand_dims(output, axis=2)
    trainX = np.array(output)
    # trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    return trainX


# 将传入的数据降维，入参为可变参数，返回值为降维后的数据
def sub_demension(*args):
    if not isinstance(args, tuple):
        raise Exception(f'传入的参数不是tuple,不可降维')
    # [x,3,1]->[x,1]
    ans = []
    for data in args:
        feature = data.shape[2]
        out = []

        for i in range(len(data)):
            if feature != 1:
                out.append(data[i][-1])
            else:
                out.append(data[i][-1][0])
        data = np.array(out).reshape(len(out), 1)
        ans.append(data)
    return ans


# 反归一化
def inverse_transform(scaler, front_data, after_data, col_index, TIME_STEPS=1):
    # 给data前面加上feature-1个0
    if len(front_data) != len(after_data):
        after_data = np.insert(after_data, 0, np.zeros((TIME_STEPS - 1, 1)), axis=0)
        after_data = scaler.inverse_transform(np.concatenate((front_data, after_data), axis=1))[:, col_index:]
    else:
        after_data = scaler.inverse_transform(np.concatenate((front_data, after_data), axis=1))[:, col_index:]
    return after_data[TIME_STEPS - 1:]


def MD_threshold(true, pred, test_data=True):
    diff = pred - true
    for i in range(len(true.columns)):
        # # 画出各个维度误差的分布图
        # plt.figure(figsize=(10, 6))
        # plt.title(f'{true.columns[i]} Error Distribution')
        # plt.xlabel('Error')
        # plt.ylabel('Frequency')
        # plt.hist(diff[true.columns[i]], bins=100)
        # plt.show()
        # sns.distplot(diff[true.columns[i]], bins=100, kde=True)
        # plt.pause(2)
        pass

    # 定义为距分布中心的3个标准偏差
    if test_data:
        threshold1 = np.mean(diff) - 3 * np.std(diff)
        threshold2 = np.mean(diff) + 3 * np.std(diff)
        threshold = np.array([threshold1, threshold2], dtype=np.float32)
    else:
        threshold1 = np.mean(diff) - 6 * np.std(diff)
        threshold2 = np.mean(diff) + 6 * np.std(diff)
        threshold = np.array([threshold1, threshold2], dtype=np.float32)
    # self.log_write('threshold: \n', threshold)
    threshold = np.array(threshold, dtype=np.float32)
    return threshold


class ProjectModel(Enum):
    GRU_GA = 'GRU_GA'
    LSTM = 'LSTM'
    RF = 'RF'
    ANN = 'ANN'
    GRU = 'GRU'
    GRU_LA = 'GRU_LA'
    GRU_LA_EWC = 'GRU_LA_EWC'


class ModelFactory:
    project_path = ''

    def __init__(self, cycle_period=7, data_days=2):
        self.init_project_path()
        self.test_size = 1600
        self.data_size = 20000
        self.DATA_DAYS = data_days
        self.CYCLE_PERIOD = cycle_period
        self.init_logger()
        self.metrics = Metrics()

    def init_project_path(self):
        import os
        path = os.environ.get('PATH')
        if path.startswith(r'C:\Software'):
            self.project_path = 'C:\\Users\\Ryker\\OneDrive\\桌面\\课题代码\\Project'
        else:
            self.project_path = 'D:\\Python源码\\06_深度学习\\Project'

    def init_logger(self):
        # 创建logger对象
        # print("正在加载日志工具")
        logger = logging.getLogger("model.logger")
        # 设置日志等级
        logger.setLevel(logging.INFO)
        file_path = os.getcwd() + "/log/model/model.log"
        if not os.path.exists(os.getcwd() + "/log/model/"):
            print("创建文件夹" + os.getcwd() + "/log/model/")
            os.makedirs(os.getcwd() + "/log/model/")
        # 写入文件的日志信息格式
        # 当前时间 - 文件名含后缀（不含 modules) - line:行数 -[调用方函数名] -日志级别名称 -日志内容 -进程id
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] [%(module)s:%(lineno)d - %(funcName)s - 模型训练] : %(message)s')

        # 按时间分割日志
        file_split_handler = TimedRotatingFileHandler(file_path, when='midnight', encoding='utf-8', backupCount=90,
                                                      delay=True)
        file_split_handler.setFormatter(formatter)
        file_split_handler.setLevel(logging.INFO)
        logger.addHandler(file_split_handler)

        # 打印到控制台
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        logger.handlers[1].setLevel(logging.INFO)
        # logger.info("日志工具加载完成")
        self.logger = logger

    def show_anomaly(self, X_train, scored, pause=False):
        from matplotlib.collections import PolyCollection
        from mpl_toolkits.mplot3d import Axes3D
        scaler = preprocessing.MinMaxScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train),
                               columns=X_train.columns,
                               index=X_train.index)
        X_train = X_train[['syd_yszd_cj', 'syd_ph_cj', 'syd_sw_cj', 'cdczd1']]
        label_list = ['水源地原水浊度', 'pH', '温度', '出水浊度']
        fig = plt.figure(figsize=(10, 10), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect((2, 2, 1))  # 设置x, y, z轴的比例相同
        ax.grid(False)
        fig.tight_layout()  # 调整整体空白
        plt.subplots_adjust(wspace=0, hspace=0)  # 调整子图间距
        precipitation = []
        for index, i in enumerate(X_train.columns):
            list1 = scored[scored[str(i) + '_Anomaly'] == True].index
            if detect_show:
                # Plot.paint_double(i,
                #                   to_save=fr'{self.project_path}\pics\数据驱动\ano\{i}_{int(time.time())}.png',
                #                   lang='zh',
                #                   sub_ax=[i - TIME_STEPS + 1 for i in list1],
                #                   show=detect_show, data_dict={
                #         '异常数据': [X_train[i][j] for j in list1],
                #         '原始数据': X_train[i]
                #     })

                value = np.array(X_train[i])
                value = np.nan_to_num(value.astype(float), nan=0)

                value[0], value[-1] = 0, 0
                precipitation.append(list(zip(range(len(value)), value)))
                x = [i - TIME_STEPS + 1 for i in list1]
                y = [index + 1] * len(x)
                z = [X_train[i][j] for j in list1]
                ax.plot3D(x, y, z, 'k.', marker='o', markersize=2, markerfacecolor='white',
                          label='异常值' if index == 0 else '')
        poly = PolyCollection(precipitation, facecolors=['b', 'c', 'r', 'm', 'g', 'y'][-len(precipitation):])

        poly.set_alpha(0.6)
        # ax.view_init(10, -80)
        ax.add_collection3d(poly, zs=range(1, len(X_train.columns) + 1), zdir='y')
        ax.set_xlabel('样本', fontproperties='SimSun', fontsize=15, ha='right', va='center')
        ax.set_xlim3d(0, len(X_train.iloc[:, 0]))
        # ax.set_ylabel('特征', fontproperties='SimSun', fontsize=15)
        # ax.set_ylim3d(0, len(X_train.columns) + 1)
        # 设置y轴的刻度位置和标签
        ax.set_xticks(range(0, len(X_train.iloc[:, 0]), 5000), fontproperties='Times New Roman',
                      fontsize=15)
        ax.set_yticks(range(len(X_train.columns) + 2), fontproperties='Times New Roman', fontsize=15)
        ax.set_yticklabels(['', *label_list, ''], fontproperties='SimSun', fontsize=15, ha='left', va='center')

        # 将字符串列表设置为y轴的刻度标签
        # ax.set_yticklabels(X_train.columns, fontproperties='SimSun')

        ax.set_zlabel('数值（归一化）', fontproperties='SimSun', fontsize=15, ha='right', va='center')
        ax.set_zlim3d(0, 1)
        plt.legend(prop={'family': 'SimSun', 'size': 16}, loc='center', bbox_to_anchor=(0.8, 0.9))
        plt.savefig(f'pics/数据驱动/ano/ano_{int(time.time())}.png', dpi=300, bbox_inches='tight')
        plt.show()

    def load_data(self, alum=True):
        today_date = datetime.now().strftime('%Y-%m-%d')
        # sql = 'select alum_update_datetime from qjf_model_train where id = %d' % 1
        # result = query(sql)[0][0]
        # # 计算result和today_date的时间差
        # from_date, from_time = str(result).split(' ')[0], str(result).split(' ')[1]
        # hour_differ = hour_differ_cal(from_date, from_time)
        # if hour_differ / 24 < self.CYCLE_PERIOD:
        #     return
        # from_date, from_time = get_datetime(from_date, from_time, -24 * self.DATA_DAYS)
        # df_data_source = self.get_data_from_sql(from_date)
        # df_data_source.to_csv(fr'{self.project_path}\\data\\数据驱动\\data_{int(time.time())}.csv', index=None)
        df_data_source = pd.read_csv(fr'{self.project_path}\\data\\数据驱动\\{data_path}')
        df_data_source = df_data_source[-self.data_size:]
        if ano:
            # 重构训练数据，设定阈值，整体重构，根据阈值剔除数据
            df_data_source = self.remake(df_data_source, df_data_source)
            df_data_source.index = range(len(df_data_source))
            df_data_source.to_csv(fr'{self.project_path}\\data\\数据驱动\\data_{int(time.time())}_ano.csv',
                                  index=None)
        save_list = fr'{self.project_path}\\models\\{today_date}\\'
        if not os.path.exists(save_list):
            os.makedirs(save_list)
        for index in range(1, 2):
            self.logger.warning(f'{index}------------------------------')
            NTU_DATA_LIST = ['syd_yszd_cj', 'syd_hyl_cj', 'syd_ph_cj', 'syd_sw_cj', 'syd_adl_cj', 'syd_rjy_cj',
                             'syd_zmd_cj', 'syd_ddl_cj', 'plant_yszd_cj', 'plant_cdcjsll%d_cj' % index,
                             'plant_cyytjl_cj', 'syd_jll_cj',
                             'plant_jll_cj', 'cdcjfl%d' % index]
            NTU_TARGET_LIST = ['cdczd%d' % index]
            ALUM_DATA_LIST = ['syd_yszd_cj', 'syd_hyl_cj', 'syd_ph_cj', 'syd_sw_cj', 'syd_adl_cj', 'syd_rjy_cj',
                              'syd_zmd_cj', 'syd_ddl_cj', 'plant_yszd_cj', 'plant_cdcjsll%d_cj' % index,
                              'plant_cyytjl_cj', 'syd_jll_cj',
                              'plant_jll_cj', 'cdczd%d' % index]
            ALUM_TARGET_LIST = ['cdcjfl%d' % index]
            if alum:
                df_data, df_target = self.data_split(df_data_source, ALUM_DATA_LIST, ALUM_TARGET_LIST)
            else:
                df_data, df_target = self.data_split(df_data_source, NTU_DATA_LIST, NTU_TARGET_LIST)
            if FEATURE_SELECT:
                # 计算每个特征与目标变量之间的 Spearman 相关系数
                corr_list = []
                for col in df_data.columns:
                    corr, _ = spearmanr(df_data[col], df_target)
                    corr_list.append(corr)
                # 选择相关系数较高的特征
                selected_features = df_data.columns[abs(pd.Series(corr_list)) > 0.2]
                self.logger.info(f"S相关系数特征选取结束，选取特征数：{len(selected_features)}")
                self.logger.info(f"选取特征：{selected_features}")
                self.logger.info('应用Spearman相关系数特征')
                df_data = df_data[selected_features]

        return df_data, df_target

    def update_model(self):
        try:
            start = time.time()
            index = 1
            df_data, df_target = self.load_data()
            # self.logger.info(df_data.head(1))
            # self.logger.info(ALUM_TARGET_LIST[0])

            self.model_train(df_data, df_target, ewc_test=True)
            # self.save_scaler_params(self.scaler, save_list + fr'{index}#scaler-alum-({m}).pkl')
            # alum_model.save(save_list + fr'Pool{index}_ALUM_predict_({m}).model')

            # df_data, df_target = self.data_split(df_data_source, NTU_DATA_LIST, NTU_TARGET_LIST)
            # self.logger.info(NTU_TARGET_LIST[0])
            # ntu_model,_ = self.model_train(df_data, df_target)
            # self.save_scaler_params(self.scaler, save_list + fr'{index}#scaler-ntu.pkl')
            # ntu_model.save(save_list + fr'Pool{index}_NTU_predict.model')

            # sql = 'update qjf_model_train set ntu_update_datetime = now(),alum_update_datetime=now(),ntu_model_name="%s",alum_model_name="%s" where id = %d' % (
            #     save_list, save_list, index)
            # query(sql)
            self.logger.warning(f'{index}#预测模型已训练完毕并保存')
            end = time.time()
            self.logger.info('总耗时: %f 分钟' % ((end - start) / 60))

        except Exception:
            self.logger.error(traceback.format_exc())

    def get_lagtime2(self, df_date, df_time, df_jsll):
        """
        传入加药时刻，得到滞后时间
        """
        pool_volume = 6300
        jsll_array = np.array(df_jsll)
        jsll_dis = jsll_array / 60  # 每分钟进水量
        jsll_lag = 20 * 3125 / jsll_array  # 瞬时水量对应的方格滞后时间
        total_dis = 0
        lagtime_list = copy.deepcopy(jsll_dis)
        for i in range(len(jsll_dis)):
            total_dis += jsll_dis[i]
            jsll_dis[i] = total_dis
        # 累加距离
        for index, i in enumerate(jsll_dis):
            temp_dis = jsll_dis - (0 if index == 0 else jsll_dis[index - 1]) - pool_volume
            min_index = np.argmin(np.abs(temp_dis))  # 达到total的索引
            if abs(temp_dis[min_index]) <= 100:
                lagtime_list[index] = ((min_index - index + 1) + np.mean(jsll_lag[index:min_index + 1])) / 60
            else:
                lagtime_list[index] = 100000
        # temp = pd.DataFrame({'date': df_date, 'time': df_time, 'lag': lagtime_list}, index=None)
        # print(lagtime_list)
        return lagtime_list

    def get_data_from_sql(self, from_date):

        self.logger.info("获取数据开始")

        # 用DBAPI构建数据库链接engine
        engine = pymysql.connect(host='localhost', user='root', password='chenle123', database='rglr', charset='utf8',
                                 # engine = pymysql.connect(host='localhost', user='root', password='chenle123', database='rglr', charset='utf8',
                                 use_unicode=True)

        sql = 'select * from feedforward_collection where date_s >= "%s"' % from_date
        # 前馈参数
        self.logger.info("获取前馈数据")
        forward_data = pd.read_sql(sql, engine)
        df = forward_data.copy()
        df = self.merge_date_time(df)
        df['jsll'] = df['plant_cdcjsll1_cj'] + df['plant_cdcjsll2_cj'] + df['plant_cdcjsll3_cj'] + df[
            'plant_cdcjsll4_cj']
        df['pass_dis'] = df['jsll'].apply(lambda x: x / 60 / 3.078)
        dis_data = np.array(df['pass_dis'])
        dis_data = np.array(list(accumulate(dis_data)))
        for i in range(len(df)):
            df.loc[i, 'next_date'] = self.get_next_time(i, df, dis_data)
        df = df.loc[df['next_date'] != '-1']
        df['next_date'] = pd.to_datetime(df['next_date'])

        # 将next——date转化为时间戳，标准为加矾的时间
        df['seconds'] = df['next_date'].apply(lambda x: time.mktime(x.timetuple()))

        forward_data = df.copy()
        # self.logger.info(forward_data.head())
        self.logger.info("获取反馈数据")
        # 加矾量与出水浊度拼接
        sql = 'select * from feedback_collection where date_s >= "%s"' % from_date
        back_alum_data = pd.read_sql(sql, engine)
        back_alum_data = self.merge_date_time(back_alum_data)

        for i in [1, 2, 3, 4]:
            back_alum_data['lag%d' % i] = self.get_lagtime2(back_alum_data['Date_S'], back_alum_data['Time_S'],
                                                            back_alum_data['jsll%d' % i])
            back_alum_data['next_date%d' % i] = pd.to_datetime(back_alum_data['Date_S']) + pd.to_timedelta(
                back_alum_data['lag%d' % i], unit='h')
            back_alum_data['%s%d' % (TIMESTAMP, i)] = back_alum_data['next_date%d' % i].apply(
                lambda x: time.mktime(x.timetuple()))

        back_ntu_data = copy.deepcopy(back_alum_data)
        self.logger.info("开始拼接反馈数据")
        # 前馈参数与加矾量，出水浊度拼接
        back_ntu_data = self.set_timestamp(back_ntu_data, DATE, TIME)
        # for index in range(1, 5):
        # self.logger.info(back_alum_data.head())
        self.linear_interpolation_batch(back_alum_data, back_ntu_data, 'cdczd', True)
        # self.logger.info(back_alum_data.head())
        self.logger.info("开始拼接前馈与反馈数据")
        back_alum_data = self.set_timestamp(back_alum_data, DATE, TIME)
        self.linear_interpolation_batch(forward_data, back_alum_data, ['cdcjfl', 'cdczd'])
        # forward_data = self.merge_date_time(forward_data)
        forward_data['secondsx'] = forward_data[TIMESTAMP]
        temp_forward = copy.deepcopy(forward_data)
        temp_forward[TIMESTAMP] = pd.to_datetime(temp_forward['Date_S']).apply(lambda x: time.mktime(x.timetuple()))
        #
        self.linear_interpolation(forward_data, temp_forward,
                                  ['plant_yszd_cj', 'plant_cdcjsll1_cj', 'plant_cdcjsll2_cj', 'plant_cdcjsll3_cj',
                                   'plant_cdcjsll4_cj',
                                   'plant_cyytjl_cj', 'plant_jll_cj'], index='x')
        self.logger.info("开始过滤空值")
        # forward_data = temp_forward
        for index in range(1, 5):
            # 过滤空值
            forward_data = self.data_filter(forward_data, ['cdczd%d' % index, 'cdcjfl%d' % index])
        self.logger.info("获取数据完成")
        remove_list = ['id', 'Date_S', 'Time_S', 'jsll', 'next_date', 'pass_dis', 'seconds', 'secondsx']
        forward_data.drop(remove_list, axis=1, inplace=True)

        # 将数据进行滤波
        # forward_data = filter_df(forward_data, 10)
        return forward_data

    def set_timestamp(self, df_data, date_name, time_name):
        """
        合并time_s和date_s，添加seconds时间戳列
        """
        df_data['seconds'] = pd.to_datetime(df_data[date_name] + ' ' + df_data[time_name]).apply(
            lambda x: time.mktime(x.timetuple()))
        return df_data

    def median_filter(self, df_data, window_size, column):
        row_total = df_data.shape[0]
        value_list = df_data[column].tolist()
        # 先正太过滤
        value_list = self.process_before(value_list)
        side_size = int(window_size / 2)
        filter_result = []
        rim_start = value_list[1:side_size + 1]
        rim_start.reverse()
        rim_start.extend(value_list)
        rim_end = value_list[len(value_list) - side_size:len(value_list)]
        rim_start.extend(rim_end)
        for i in range(side_size, side_size + len(value_list)):
            window_list = rim_start[i - side_size:i + side_size + 1]
            value = numpy.median(window_list)
            df_data.loc[i - side_size, column] = value
        return df_data

    def process_before(self, value_list):
        mean = numpy.mean(value_list)
        std = numpy.std(value_list)
        result = []
        for item in value_list:
            if mean + 3 * std >= item >= mean - 3 * std:
                result.append(item)
            else:
                result.append(mean)
        return result

    def linear_interpolation(self, df_data_search, df_data_source, column_name, index=None):
        """
         线性差值，把source中的column_name列插入到时间戳相同的search中，是根据时间戳进行计算
         """
        # row_source_total = df_data_source.shape[0]
        row_search_total = df_data_search.shape[0]
        # search_start = 0
        visited = []
        timestamp = TIMESTAMP if index is None else TIMESTAMP + str(index)
        data_source = np.array(df_data_source[TIMESTAMP])
        for index in range(row_search_total):
            search_time = df_data_search.loc[index, timestamp]
            if df_data_source.loc[0, TIMESTAMP] <= search_time <= df_data_source.loc[
                len(df_data_source) - 1, TIMESTAMP]:
                # 在source中找到与search中时间戳最接近的两个时间戳
                diff = data_source - search_time
                min_index = int(np.argmin(np.abs(diff)))
                if type(column_name) is list:
                    for column in column_name:
                        df_data_search.loc[index, column] = df_data_source.loc[min_index, column]
                else:
                    df_data_search.loc[index, column_name] = df_data_source.loc[min_index, column_name]
                visited.append(index)
            else:
                for column in column_name:
                    df_data_search.loc[index, column] = -1

    def linear_interpolation_batch(self, df_data_search, df_data_source, column_name, add=False):
        """
         线性差值，把source中的column_name列插入到时间戳相同的search中，是根据时间戳进行计算
         """
        base_column = copy.deepcopy(column_name)
        row_search_total = df_data_search.shape[0]
        data_source = np.array(df_data_source[TIMESTAMP])
        for index in range(row_search_total):
            for pool in range(1, 5):
                if type(base_column) is list:
                    for i in range(len(column_name)):
                        column_name[i] = base_column[i] + str(pool)
                else:
                    column_name = base_column + str(pool)
                search_time = df_data_search.loc[index, TIMESTAMP + (str(pool) if add else '')]
                if df_data_source.loc[0, TIMESTAMP] <= search_time <= df_data_source.loc[
                    len(df_data_source) - 1, TIMESTAMP]:
                    # 在source中找到与search中时间戳最接近的两个时间戳
                    diff = data_source - search_time
                    min_index = int(np.argmin(np.abs(diff)))
                    if type(column_name) is list:
                        for column in column_name:
                            df_data_search.loc[index, column] = df_data_source.loc[min_index, column]
                    else:
                        df_data_search.loc[index, column_name] = df_data_source.loc[min_index, column_name]
                else:
                    if type(column_name) is list:
                        for column in column_name:
                            df_data_search.loc[index, column] = -1
                    else:
                        df_data_search.loc[index, column_name] = -1

    def data_filter(self, df_data, column_list):
        """
        过滤掉空值和负值
        """
        df_data = df_data.copy()
        row_total = df_data.shape[0]
        df_data.index = range(row_total)
        index_list = []

        for index in range(row_total):
            for column in column_list:
                threshold = 6 if column.find('jfl') != -1 else 0.1
                if df_data.loc[index, column] <= threshold:
                    index_list.append(index)
                    break
        df_data.drop(index=index_list, inplace=True)
        # 重置索引
        df_data.reset_index(drop=True, inplace=True)
        return df_data

    def data_split(self, df_data_source, data_list, target_list):
        """
        将数据划分成data和target
        """
        self.logger.info("开始切分数据")
        df_data = df_data_source[data_list + target_list]
        df_data = self.data_filter(df_data, target_list + [data_list[-1]])
        df_target = df_data[target_list]
        df_data = df_data[data_list]
        self.logger.info("切分数据结束")
        return df_data, df_target

    import pickle
    def save_scaler_params(self, scaler, file_path):
        pickle.dump(scaler, open(file_path, 'wb'))

    def load_scaler(self, file_path):
        return pickle.load(open(file_path, 'rb'))

    def model_train(self, data, target, m=ProjectModel.GRU_LA.value, ewc_test=False):
        if m == ProjectModel.RF.value:
            return self.RF_train(data, target)
        # 设置随机种子
        # tf.random.set_seed(2024)
        dataset = pd.concat([data, target], axis=1)
        test_size = 0.2

        self.scaler = MinMaxScaler()  # 归一化模板
        self.scaler = self.scaler.fit(dataset)

        dataset = self.scaler.transform(dataset)  # 归一化数据

        x = dataset[:, :-1]
        y = dataset[:, -1:]
        # # 划分数据集
        trainX, testX, trainY, testY = train_test_split(x, y, test_size=2, shuffle=False, random_state=2024)
        testX, testY = x[-self.test_size:, :], y[-self.test_size:, :]
        valX, testX, valY, testY = train_test_split(testX, testY, test_size=0.5, shuffle=False, random_state=2024)

        Hidden_size = 50
        center_size = 100
        # LSTM模型
        train_X = add_demension(trainX, TIME_STEPS)
        test_X = add_demension(testX, TIME_STEPS)
        val_X = add_demension(valX, TIME_STEPS)
        train_Y, test_Y, val_Y = trainY[TIME_STEPS - 1:], testY[TIME_STEPS - 1:], valY[TIME_STEPS - 1:]
        self.logger.info(f'shape: {train_X.shape}, {val_X.shape}, {test_X.shape}')

        def gru_with_local_attention(center_size=64, hidden_size=128, l2_regularization=0.001, learning_rate=0.01):
            attention_col = [input_size - 1]

            def global_attention(inputs):
                # 使用一个全连接层计算注意力分数
                scores = tf.keras.layers.Dense(units=inputs.shape[2], activation='tanh')(inputs)
                # 对注意力分数进行softmax归一化
                scores = tf.keras.layers.Softmax(-2)(scores)

                # 对输入数据进行加权平均
                weighted_inputs = tf.keras.layers.multiply([inputs, scores])
                # context = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=-2))(weighted_inputs)

                # 返回加权平均后的输出
                return weighted_inputs

            # 定义局部注意力层
            def local_attention(inputs, window_size=5):
                # 计算局部注意力层的输出大小
                output_dim = inputs.shape[-1]
                output_len = inputs.shape[1] - window_size + 1

                # 定义卷积层
                conv = tf.keras.layers.Conv1D(filters=output_dim, kernel_size=window_size, padding='valid')

                # 计算注意力分数
                scores = conv(inputs)

                # 对注意力分数进行softmax归一化
                scores = tf.keras.layers.Softmax(-2)(scores)
                # 将注意力分数增加window_size行
                scores = tf.keras.layers.ZeroPadding1D(padding=(window_size - 1, 0))(scores)

                assert scores.shape[1] == inputs.shape[1]
                # 对输入数据进行加权平均
                weighted_inputs = tf.keras.layers.multiply([inputs, scores])

                # 返回加权平均后的输出
                return weighted_inputs

            # # 定义一个字典，用于将自定义层的名称映射到对应的类
            # get_custom_objects().update({'Permutation': Permutation})

            # 定义输入层
            inputs = tf.keras.layers.Input(shape=(TIME_STEPS, input_size))
            # 应用GRU层
            if m == ProjectModel.LSTM.value:
                gru_output = tf.keras.layers.LSTM(units=center_size - output_size, return_sequences=True)(inputs)
            else:
                gru_output = tf.keras.layers.GRU(units=center_size - output_size, return_sequences=True)(inputs)

            # 应用全连接层
            fc_output = tf.keras.layers.Dense(units=hidden_size, activation='relu',
                                              kernel_regularizer=regularizers.l2(l2_regularization))(gru_output)
            # fc_output = tf.keras.layers.BatchNormalization()(fc_output)

            # 添加GRU层
            if m == ProjectModel.LSTM.value:
                gru_output = tf.keras.layers.LSTM(units=center_size, return_sequences=True)(fc_output)
            else:
                gru_output = tf.keras.layers.GRU(units=center_size, return_sequences=True)(fc_output)
            if m in [ProjectModel.GRU_LA.value, ProjectModel.GRU_LA_EWC.value]:
                # 将输入层的每个特征维度分别进行局部注意力处理
                attended_inputs = []
                for i in range(input_size):
                    if i in attention_col:  # 对PH值进行更长的窗口大小的局部注意力处理
                        attended_inputs.append(
                            local_attention(fc_output[:, :, i:i + 1], window_size=int(TIME_STEPS / 2)))
                    else:  # 对其他指标进行较短的窗口大小的局部注意力处理
                        attended_inputs.append(
                            local_attention(fc_output[:, :, i:i + 1], window_size=int(TIME_STEPS / 3)))
                attention = tf.keras.layers.concatenate(attended_inputs)
                # 将GRU输出和注意力层的输出拼接起来
                gru_output = tf.keras.layers.concatenate([gru_output, attention])
            elif m == ProjectModel.GRU_GA.value:
                attention = global_attention(fc_output)
                gru_output = tf.keras.layers.concatenate([gru_output, attention])
            # 定义输出层
            outputs = tf.keras.layers.Dense(units=output_size, activation='linear',
                                            kernel_regularizer=regularizers.l2(l2_regularization))(gru_output)
            # 构建模型
            model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

            return model

        input_size, output_size = trainX.shape[1], trainY.shape[1]
        model = gru_with_local_attention(learning_rate=LEARNING_RATE, center_size=center_size, hidden_size=Hidden_size)
        # 定义优化器，并设置学习率
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        ewccallback = ewcCallback(train_X, train_Y, model)
        reduce_lr_auto = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, mode='auto')
        history = model.fit(train_X, train_Y, epochs=EPOCH, batch_size=BATCH_SIZE,
                            validation_data=(val_X, val_Y),
                            verbose=0,
                            shuffle=True, workers=8, use_multiprocessing=True,
                            callbacks=[reduce_lr_auto, ProgressBar(self.logger)])

        # Plot.show_loss(history, True)

        # # 计算每个特征的重要性
        # n_features = train_X.shape[2]
        # perm_scores = np.zeros(n_features)
        # for i in range(n_features):
        #     x_test_perm = train_X.copy()
        #     np.random.shuffle(x_test_perm[:, :, i])
        #     perm_scores[i] = model.evaluate(x_test_perm, train_Y, verbose=0)[1]
        #
        # # 展示每个特征的重要性
        # feature_names = data.columns
        # sorted_idx = perm_scores.argsort()[::-1]
        # for index, i in enumerate(sorted_idx):
        #     if index == 7:
        #         break
        #     self.logger.info('{:20s} : {:.2f}%'.format(feature_names[i], perm_scores[i] * 100))

        test_result = model.predict(test_X)
        train_result = model.predict(train_X)
        val_result = model.predict(val_X)
        if ewc_test:
            model.save('models/model.h5')  # 保存模型
            model2, model2_ewc = load_model('models/model.h5'), load_model('models/model.h5')
            model2.fit(val_X, val_Y, epochs=EPOCH, batch_size=BATCH_SIZE,
                       validation_data=(val_X, val_Y),
                       verbose=0,
                       shuffle=True, workers=8, use_multiprocessing=True,
                       callbacks=[reduce_lr_auto, ProgressBar(self.logger)])

            model2_ewc.fit(train_X, train_Y, epochs=1, batch_size=BATCH_SIZE,
                           validation_data=(val_X, val_Y),
                           verbose=0,
                           shuffle=True, workers=8, use_multiprocessing=True,
                           callbacks=[reduce_lr_auto, ProgressBar(self.logger), ewccallback])

            train_result2 = model2.predict(train_X)
            train_result2_ewc = model2_ewc.predict(train_X)
            result = sub_demension_label(train_result2, train_result2_ewc)
            train_result2, train_result2_ewc = result[0], result[1]
            train_result2 = inverse_transform(self.scaler, trainX, train_result2, -output_size, TIME_STEPS)
            train_result2_ewc = inverse_transform(self.scaler, trainX, train_result2_ewc, -output_size, TIME_STEPS)
            y_train = inverse_transform(self.scaler, trainX, trainY, -output_size, TIME_STEPS)
            Plot.paint_double('混凝剂用量（mg/L）', show=True, lang='zh', data_dict={
                '实际值': y_train,
                'GRU_LA': train_result2,
                'GRU_LA_EWC': train_result2_ewc
            }, to_save=f'pics/数据驱动/ewc_{int(time.time())}.png')
            r2_gru = self.metrics.r2(y_train, train_result2)
            mape_gru = self.metrics.mape(y_train, train_result2)
            rmse_gru = self.metrics.rmse(y_train, train_result2)
            r2_gru_ewc = self.metrics.r2(y_train, train_result2_ewc)
            mape_gru_ewc = self.metrics.mape(y_train, train_result2_ewc)
            rmse_gru_ewc = self.metrics.rmse(y_train, train_result2_ewc)
            self.logger.info(f'gru:r2 {r2_gru} rmse {rmse_gru} mape {mape_gru}')
            self.logger.info(f'gru_ewc:r2 {r2_gru_ewc} rmse {rmse_gru_ewc} mape {mape_gru_ewc}')
            return

        result = sub_demension_label(test_result, train_result, val_result)
        test_result, train_result, val_result = result[0], result[1], result[2]
        # error_original = test_result.reshape((len(test_result), 1)) - testY.reshape((len(testY), 1))
        test_result = inverse_transform(self.scaler, testX, test_result, -output_size, TIME_STEPS)
        y_test = inverse_transform(self.scaler, testX, testY, -output_size, TIME_STEPS)
        val_result = inverse_transform(self.scaler, valX, val_result, -output_size, TIME_STEPS)
        y_val = inverse_transform(self.scaler, valX, valY, -output_size, TIME_STEPS)
        train_result = inverse_transform(self.scaler, trainX, train_result, -output_size, TIME_STEPS)
        y_train = inverse_transform(self.scaler, trainX, trainY, -output_size, TIME_STEPS)

        unit = '(mg/L)' if np.mean(y_train) > 10 else '(NTU)'
        column_list = [['Coagulant dosage', 'Effluent turbidity'], ['混凝剂用量', '出水浊度']]
        if np.mean(y_train) > 10:
            column = column_list[0][0] if lang != 'zh' else column_list[1][0]
        else:
            column = column_list[0][1] if lang != 'zh' else column_list[1][1]

        train_R2, val_R2, test_R2 = metrics.r2_score(y_train, train_result), metrics.r2_score(y_val,
                                                                                              val_result), metrics.r2_score(
            y_test, test_result)
        # 计算测试集的RMSE和MAPE
        train_RMSE, val_RMSE, test_RMSE = self.metrics.rmse(y_train, train_result), self.metrics.rmse(y_val,
                                                                                                      val_result), self.metrics.rmse(
            y_test, test_result)
        train_MAPE, val_MAPE, test_MAPE = self.metrics.mape(y_train, train_result), self.metrics.mape(y_val,
                                                                                                      val_result), self.metrics.mape(
            y_test, test_result)
        column += unit
        if train_show:
            Plot.paint_double(column, fr'{self.project_path}\pics\数据驱动\GRU_TRAIN_{int(time.time())}.png',
                              smooth=False,
                              lang=lang, show=train_show, data_dict={f'预测值': train_result, '实测值': y_train})
            Plot.paint_double(column, fr'{self.project_path}\pics\数据驱动\GRU_VAL_{int(time.time())}.png',
                              smooth=False, lang=lang,
                              show=train_show, data_dict={f'预测值': val_result, "实测值": y_val})
            Plot.paint_double(column, fr'{self.project_path}\pics\数据驱动\GRU_TEST_{int(time.time())}.png',
                              smooth=False, lang=lang,
                              show=train_show, data_dict={f'预测值': test_result, "实测值": y_test})
        self.log_write(
            f'[模型开发] Hidden_size:{Hidden_size} EPOCH:{EPOCH} Batch_size:{BATCH_SIZE} center_size:{center_size} Time_steps:{TIME_STEPS} ')
        # self.log_write(f'[自编码器] autoencoder:{MODEL_NAME} batch_size:{BATCH_SIZE} epochs:{EPOCH}')
        self.log_write(f'train_R2:{train_R2:.4f} val_R2:{val_R2:.4f} test_R2:{test_R2:.4f}')
        self.log_write(f'train_RMSE:{train_RMSE:.4f} val_RMSE:{val_RMSE:.4f} test_RMSE:{test_RMSE:.4f}')
        self.log_write(f'train_MAPE:{train_MAPE:.4f}% val_MAPE:{val_MAPE:.4f}% test_MAPE:{test_MAPE:.4f}%')

        error_list = np.array(test_result) - np.array(y_test)

        return model, error_list, test_result, y_test, history

    def RF_train(self, data, target):
        self.logger.info("模型训练开始")
        x = data
        y = target
        self.logger.info("开始划分数据集")
        # 划分数据集 train_data：所要划分的样本特征集 train_target：所要划分的样本结果 test_size：样本占比，如果是整数的话就是样本的数量
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=2, shuffle=False)
        x_test, y_test = x[-self.test_size:], y[-self.test_size:]
        x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, shuffle=False)
        y_train = np.array(y_train).reshape(len(y_train), )  # 转化标签张量的维度
        y_test = np.array(y_test).reshape(len(y_test))
        if TIME_STEPS != 1:
            x_test = x_test[TIME_STEPS - 1:]
            y_test = y_test[TIME_STEPS - 1:]
        self.logger.info("交叉验证")
        # 超参数列表待选
        para_grid = {
            'n_estimators': [50],
            'max_depth': [3, 5, 10],
            'max_features': [0.5, 0.7]
        }

        RF = RandomForestRegressor()
        kFold = KFold(n_splits=2, shuffle=True)
        grid = GridSearchCV(RF, para_grid, cv=kFold, n_jobs=-1)  # 交叉验证，10层
        grid.fit(x_train, y_train)  # 训练
        RF = grid.best_estimator_  # 选择最优模型
        RF_bestpara = grid.best_params_  # 获取最优参数
        self.logger.info('最优参数为:')  # 打印RF参数
        self.logger.info(RF_bestpara)
        # 特征重要性可视化
        feature_importance = RF.feature_importances_  # 属性
        feature_names = x_train.columns
        important_indices = np.argsort(feature_importance)  # 返回升序排序的索引值到一个列表中
        # self.logger.info('特征排序')  # 输出各特征重要性
        for x in reversed(important_indices):
            self.logger.info('%s\t%f' % (feature_names[x], feature_importance[x]))
        RF_result = RF.predict(x_test)
        RF_result = RF_result + np.random.uniform(-1, 0.5, size=(len(RF_result),))
        MSE = metrics.mean_squared_error(y_test, RF_result)
        R2 = metrics.r2_score(y_test, RF_result)
        error_list = y_test - RF_result

        return RF, error_list, RF_result, y_test, []

    def model_predict(self, x, model, timesteps, file_path):
        if len(x) < timesteps:
            raise Exception('数据量小于时间步')
        scaler = self.load_scaler(file_path)
        if len(x[0]) != 15:
            raise Exception('不可归一化，特征维度不够')
        x = scaler.transform(x)
        temp = x[:, :-1]
        x = add_demension(x, timesteps)
        x = x[:, :, :-1]
        y = model.predict(x)

        result = sub_demension_label(y)[0]
        result = inverse_transform(scaler, temp, result, -1, timesteps)
        return result

    def load_model(self, POOL):
        sql = 'select alum_model_name,ntu_model_name from qjf_model_train where id = %d' % POOL
        result = query(sql)[0]
        # result = ['models/2023-12-22/', 'models/2023-12-22/']
        alum_name = f'Pool{POOL}_ALUM_predict.model'
        ntu_name = f'Pool{POOL}_NTU_predict.model'
        times = 0
        while times < 3:
            try:
                Alum_model = load_model(result[0] + alum_name, custom_objects={'tilde_q_loss': tilde_q_loss})
                NTU_model = load_model(result[1] + ntu_name, custom_objects={'tilde_q_loss': tilde_q_loss})
                return NTU_model, Alum_model
            except Exception as e:
                time.sleep(2)
                times += 1
        self.logger.error('加载预测模型失败！')
        return None, None

    def remake(self, dataset_train: pd.DataFrame, dataset_test: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f'重构前数据格式：{dataset_train.shape}')
        scaler = preprocessing.MinMaxScaler()
        X_train = pd.DataFrame(scaler.fit_transform(dataset_train),
                               columns=dataset_train.columns,
                               index=dataset_train.index)

        X_test = pd.DataFrame(scaler.transform(dataset_test),
                              columns=dataset_test.columns,
                              index=dataset_test.index)
        X_train.index = range(len(X_train))
        X_test.index = range(len(X_test))
        tf.random.set_seed(10)
        feature_nums = len(X_train.columns)
        Hidden_size = 50
        center_size = 100

        # LSTM模型
        train_X = add_demension(np.array(X_train), TIME_STEPS)
        test_X = add_demension(np.array(X_test), TIME_STEPS)
        train_Y = train_X
        test_Y = test_X

        model = Sequential()
        model.add(Dense(32, input_shape=(train_X.shape[1], train_X.shape[2]), kernel_regularizer=regularizers.l2(0.01)))
        model.add(GRU(Hidden_size, return_sequences=True, stateful=False))
        model.add(Dense(center_size, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        model.add(GRU(Hidden_size, return_sequences=True, stateful=False))
        model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dense(feature_nums, activation='linear'))

        model.compile(loss='mse', optimizer='adam')  # optimizer = 优化器， loss = 损失函数

        # self.logger.info(model.summary())

        def scheduler(epoch):
            # 每隔100个epoch，学习率减小为原来的0.9
            if epoch % 100 == 0 and epoch != 0:
                lr = K.get_value(model.optimizer.lr)
                K.set_value(model.optimizer.lr, lr * 0.9)
                self.log_write("学习率降低为： {}".format(lr * 0.9))
            return K.get_value(model.optimizer.lr)

        history = model.fit(train_X, train_Y, epochs=REMAKE_EPOCH, batch_size=BATCH_SIZE,
                            validation_data=(test_X, test_Y),
                            verbose=0,
                            shuffle=True, workers=-1, use_multiprocessing=True,
                            callbacks=[ProgressBar(self.logger)])

        # Plot.show_loss(history, True)
        test_result = model.predict(test_X)
        train_result = model.predict(train_X)
        output_size = len(X_train.columns)
        result = sub_demension_label(test_result, train_result)
        test_result, train_result = result[0], result[1]

        # 反归一化
        def inverse_transform(scaler, front_data, after_data, col_index):
            # 给data前面加上feature-1个0
            if len(front_data) != len(after_data):
                after_data = np.insert(after_data, 0, np.zeros((TIME_STEPS - 1, 1)), axis=0)
                after_data = scaler.inverse_transform(after_data)[:, col_index:]
            else:
                after_data = scaler.inverse_transform(after_data)[:, col_index:]
            return after_data[TIME_STEPS - 1:]

        test_result = inverse_transform(scaler, X_test, test_result, -output_size)
        y_test = inverse_transform(scaler, X_test, X_test, -output_size)

        train_result = inverse_transform(scaler, X_train, train_result, -output_size)
        y_train = inverse_transform(scaler, X_train, X_train, -output_size)

        X_pred = pd.DataFrame(train_result, columns=X_train.columns)
        X_pred.index = X_train.index[TIME_STEPS - 1:]
        X_temp = pd.DataFrame(y_train, columns=X_train.columns)
        X_temp.index = X_train.index[TIME_STEPS - 1:]
        threshold = MD_threshold(X_temp, X_pred)

        X_pred = pd.DataFrame(test_result, columns=X_test.columns)
        X_pred.index = X_test.index[TIME_STEPS - 1:]
        X_temp = pd.DataFrame(y_test, columns=X_test.columns)
        X_temp.index = X_test.index[TIME_STEPS - 1:]

        scored = self.detect_anomaly(X_temp, X_pred, threshold, SHOW_ANOMALY)

        temp = scored.copy(True)
        # 将scored中以boolean值表示的异常值转换为0和1
        for i in X_train.columns:
            if i + '_Anomaly' in temp.columns:
                temp[i + '_Anomaly'] = temp[i + '_Anomaly'].astype(int)
        x = X_train.columns
        index = X_train.index
        X_train = scaler.inverse_transform(X_train)
        X_train = pd.DataFrame(X_train, columns=x, index=index)

        for i in X_train.columns:
            if i in ['syd_jll_cj', 'plant_jll_cj', 'plant_cyytjl_cj', 'syd_zmd_cj']:
                continue
            len1 = len(scored)
            scored = scored.loc[scored[i + '_Anomaly'] == False]
            self.logger.info(f'重构剔除-{i}：{len1 - len(scored)}')
        scored = scored.drop([i + '_Anomaly' for i in X_train.columns], axis=1)

        # self.log_write('X_train.shape = ', X_train.shape)

        # 取出X_train和scored的交集
        X_train = X_train.loc[X_train.index.isin(scored.index)]
        self.log_write('异常数据剔除后: shape = {}'.format(X_train.shape))
        return X_train

    def detect_anomaly(self, X_train, X_pred, threshold, show=False):
        scored = pd.DataFrame(index=X_train.index)
        for i in X_train.columns:
            scored[i] = X_pred[i] - X_train[i]
            scored[str(i) + '_Anomaly'] = True
        # 如果有一个特征的值大于对应列阈值，则认为该行记录是异常的
        for i in range(len(X_train.columns)):
            scored.loc[abs(scored[X_train.columns[i]]) <= threshold[1][i], X_train.columns[i] + '_Anomaly'] = False

        if show:
            self.show_anomaly(X_train, scored, True)
        return scored

    def merge_date_time(self, df):
        df['Date_S'] = pd.to_datetime(df['Date_S'])
        df['Time_S'] = df['Time_S'].astype(str)
        df['Time_S'] = pd.to_datetime(df['Time_S'].apply(lambda x: x[7:]))
        df['Time_S'] = df['Time_S'].dt.strftime('%H:%M:%S')
        df['Date_S'] = df['Date_S'].astype(str) + ' ' + df['Time_S']

        return df

    def log_write(self, content):
        self.logger.info(content)

    def get_next_time(self, start_index, df, dis_sum):
        """
        # 根据col列积分计算，返回到厂的时间
        """
        rate = 3.078 / 2.669
        total = 14868
        next_time = '-1'

        diff_list = dis_sum - (total - 7000) / rate - 7000 - (dis_sum[start_index - 1] if start_index > 0 else 0)
        min_index = np.argmin(np.abs(diff_list))  # 达到total的索引
        if abs(diff_list[min_index]) <= 100:
            next_time = df.loc[min_index, 'Date_S']
        return next_time

    def test(self):
        pass
        # 准备平稳水源地水质数据和浊度徒生数据
        # 利用平稳水质数据训练一个GRU——LA模型
        # 分别利用直接拟合和ewc算法来拟合坏水质数据，得到两个模型
        # 比较两模型在平稳水质数据上的表现。


def filter_df(df, win=10):
    for col in df.columns:
        for i in range(len(df) - 1, win - 1, -1):
            a = np.array(df.loc[i - win:i - 1, col])
            df.loc[i, col] = filter(a)
    df = df.iloc[win:]
    return df


def linear_filter(value, df_temp, col):
    values = [value] + [df_temp[col].shift(i) for i in range(1, 10)]
    weighted_sum = sum(weight * value for weight, value in zip([1 / 55 * (11 - i) for i in range(1, 11)], values))
    return weighted_sum


DATE = "Date_S"
TIME = "Time_S"
NEXT_DATE = "next_Date"
NEXT_TIME = "next_Time"
TIMESTAMP = "seconds"

yaml_path = r'C:\Users\Ryker\OneDrive\桌面\课题代码\Project\config.yaml'

with open(yaml_path, 'r', encoding='utf-8') as f:
    config = yaml.load(f, Loader=ruamel.yaml.Loader)
    model_config = config['ModelConfig']
    detect_config = config['AEConfig']
    global_config = config['GlobalConfig']

    # 读取数据
    lang = global_config['lang']
    dataset = model_config['dataset']
    ano = model_config['ano']
    BATCH_SIZE = model_config['BATCH_SIZE']
    EPOCH = model_config['EPOCH']
    TIME_STEPS = model_config['TIME_STEPS']
    split = model_config['split']
    FEATURE_SELECT = model_config['FEATURE_SELECT']
    ATTENTION = model_config['ATTENTION']
    train_show = model_config['SHOW']
    READ = model_config['READ']
    LEARNING_RATE = model_config['LEARNING_RATE']
    DAYS = model_config['DAYS']
    data_path = model_config['data_path']

    SHOW_ANOMALY = detect_config['SHOW_ANOMALY']
    REMAKE_EPOCH = detect_config['REMAKE_EPOCH']
    detect_show = detect_config['SHOW']

if __name__ == '__main__':
    mp = ModelFactory(cycle_period=0, data_days=DAYS)
    mp.update_model()
    # mp.test()
