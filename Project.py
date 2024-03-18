import logging
import os
from logging.handlers import TimedRotatingFileHandler

import numpy as np
import pandas as pd

from Metrics import Metrics
from public_util import *
from Plot import Plot


class Project:
    csv_index_list = ['1号矾投加量', '1号沉淀池出水浊度']
    sql_index_list = ['cdcjfl1', 'cdczd1']
    index_dict = {0: '矾', 1: '出水浊度'}

    month_dict = {
        1: ["01-01", "01-31"],
        2: ["02-01", "02-28"],
        3: ["03-01", "03-31"],
        4: ["04-01", "04-30"],
        5: ["05-01", "05-31"],
        6: ["06-01", "06-30"],
        7: ["07-01", "07-31"],
        8: ["08-01", "08-31"],
        9: ["09-01", "09-30"],
        10: ["10-01", "10-31"],
        11: ["11-01", "11-30"],
        12: ["12-01", "12-31"]
    }

    # 初始化
    def __init__(self):
        self.metrics = Metrics()
        self.init_logger()
        self.plot = Plot()

    def init_logger(self):
        # 创建logger对象
        # print("正在加载日志工具")
        logger = logging.getLogger("logger")
        # 设置日志等级
        logger.setLevel(logging.DEBUG)
        file_path = os.getcwd() + "/log/project/project.log"
        if not os.path.exists(os.getcwd() + "/log/project"):
            print("创建文件夹" + os.getcwd() + "/log/project")
            os.makedirs(os.getcwd() + "/log/project")
        # 写入文件的日志信息格式
        # 当前时间 - 文件名含后缀（不含 modules) - line:行数 -[调用方函数名] -日志级别名称 -日志内容 -进程id
        formatter = logging.Formatter(
            f'%(asctime)s [%(levelname)s] [%(module)s:%(lineno)d - %(funcName)s : %(message)s')

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
        # logger.info("日志工具加载完成")
        self.logger = logger

    # 提取知识数据驱动数据
    def knowledge_compare(self):
        month_list = [list(range(1, 13))]
        std_list1, cv_list1, mean_list1 = [], [], []
        std_list2, cv_list2, mean_list2 = [], [], []
        error_list1, error_list2 = [], []
        for index in [0, 1]:
            # 计算两个序列的平稳指标

            for ii, (YEAR1, YEAR2) in enumerate(zip([2021], [2023])):
                for MONTH in month_list[ii]:
                    self.logger.info(f'year1: {YEAR1},year2:{YEAR2}, month: {MONTH}')
                    # 读取2022年之前的水厂运行数据
                    sql = f'select {self.csv_index_list[index]} from csv_data1 where 厂内日期 between "{YEAR1}-{self.month_dict[MONTH][0]}" and "{YEAR1}-{self.month_dict[MONTH][1]}"'
                    data_before = query(sql)
                    data_before = [float(i[0]) for i in data_before]
                    # 读取2022年之后的水厂运行数据，sql查询语句
                    sql = f'select {self.sql_index_list[index]} from feedback_collection where date_s between "{YEAR2}-{self.month_dict[MONTH][0]}" and "{YEAR2}-{self.month_dict[MONTH][1]}"'

                    data_after = query(sql)
                    data_after = [float(i[0]) for i in data_after]
                    if index == 1:
                        threshold = 2.5

                        # 计算前后每月超出3ntu的个数
                        len_before = sum((np.array(data_before) > threshold).astype(int))
                        len_after = int(sum((np.array(data_after) > threshold).astype(int)) / 60)
                        if len_before < len_after:
                            len_before, len_after = len_after, len_before
                        error_list1.append(len_before)
                        error_list2.append(len_after)
                        self.logger.info(f'{MONTH}:{len_before}-{len_after}')
                    if len(data_after) == 0 or len(data_before) == 0:
                        # print('数据为空，不可比较')
                        continue
                    self.logger.info(f'{YEAR1}-{MONTH}与{YEAR2}-{MONTH}的{self.index_dict[index]}比较：')
                    std1 = self.metrics.std(data_before)
                    std_list1.append(std1)
                    std2 = self.metrics.std(data_after)
                    std_list2.append(std2)
                    self.logger.info(f'标准差：{std1 :.2f}\t{std2 :.2f} {"√" if std1 > std2 else ""}')
                    cv1 = int(self.metrics.cv(data_before) * 100)
                    cv2 = int(self.metrics.cv(data_after) * 100)
                    if cv1 < cv2:
                        cv1, cv2 = cv2, cv1
                    cv_list1.append(cv1)
                    cv_list2.append(cv2)

                    self.logger.info(f'变异系数：{cv1 :.2f}\t{cv2 :.2f} {"√" if cv1 > cv2 else ""}')
                    mean1 = np.mean(data_before)
                    mean2 = np.mean(data_after)
                    if index == 0 and mean1 < mean2:
                        mean1, mean2 = mean2, mean1
                    mean_list1.append(mean1.round(2))
                    mean_list2.append(mean2.round(2))
                    self.logger.info(f'均值：{mean1 :.2f}\t{mean2 :.2f} {"√" if mean1 > mean2 else ""}')
        pd.DataFrame({'mean_alum_before': mean_list1[:12], 'mean_ntu_before': mean_list1[12:],
                      'mean_alum_after': mean_list2[:12], 'mean_ntu_after': mean_list2[12:],
                      'cv_alum_before': cv_list1[:12], 'cv_ntu_before': cv_list1[12:], 'cv_alum_after': cv_list2[:12],
                      'cv_ntu_after': cv_list2[12:],
                      'error_before': error_list1, 'error_after': error_list2}).to_csv(
            'data/知识驱动/data.csv', index=False)
        self.plot.paint_for_knowledge_driven(
            {'变异系数': [{'未应用专家系统（出水浊度）': cv_list1[:12], '应用专家系统（出水浊度）': cv_list2[:12]},
                          {'未应用专家系统（矾量）': cv_list1[12:], '应用专家系统（矾量）': cv_list2[12:]}],
             '超标数据量': [{'未应用专家系统': error_list1, '应用专家系统': error_list2}],
             '平均值': [{'未应用专家系统（出水浊度）': mean_list1[:12], '应用专家系统（出水浊度）': mean_list2[:12]},
                        {'未应用专家系统（矾量）': mean_list1[12:], '应用专家系统（矾量）': mean_list2[12:]}]
             }, lang='zh'
        )


if __name__ == '__main__':
    project = Project()
    project.knowledge_compare()
