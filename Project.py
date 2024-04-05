import logging
import os
from logging.handlers import TimedRotatingFileHandler

import pandas as pd

from Metrics import Metrics
from public_util import *
from Plot import Plot
from model_process import ModelFactory, ProjectModel


class Project:
    csv_index_list = ['1号矾投加量', '1号沉淀池出水浊度']
    sql_index_list = ['cdcjfl1', 'cdczd1']
    index_dict = {0: '矾', 1: '出水浊度'}
    logger = None

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
        self.model_factory = ModelFactory()

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
             }, lang='zh')

    def GRU_LA_show(self, alum=True):
        lang = 'zh'
        df_data, df_target = self.model_factory.load_data(alum)
        threshold_dict = {
            True: 20,
            False: 2
        }
        alum_model, error_list_grula, result_grula, y_test, h_grula = self.model_factory.model_train(df_data, df_target)
        del_index = np.where(abs(error_list_grula) > threshold_dict[alum])
        y_test = np.delete(y_test, del_index)
        result_grula = np.delete(result_grula, del_index)

        fig = plt.figure(figsize=(12, 9), dpi=100)  # 设置画布大小，像素
        fig.subplots_adjust(top=0.85)
        # ax1显示y1  ,ax2显示y2
        ax11 = fig.add_subplot(211)
        # ax11.set_ylim(10, 200)
        ax11.tick_params(axis='x', labelsize=20)  # 设置字体大小为10
        ax11.tick_params(axis='y', labelsize=20)  # 设置字体大小为10
        ax11.set_ylabel('损失（TILDE-Q）', fontsize=20,
                        fontproperties='SimSun' if lang == 'zh' else 'Times New Roman')
        ax11.set_xlabel('回合', fontsize=20, fontproperties='SimSun' if lang == 'zh' else 'Times New Roman')
        ax11.plot(range(len(h_grula.history['loss'])), h_grula.history['loss'], color_list[:2][0] + '-',
                  label='训练集')
        ax11.plot(range(len(h_grula.history['val_loss'])), h_grula.history['val_loss'], color_list[:2][1] + '-',
                  label='验证集')

        # 获取图例对象
        handles, labels = ax11.get_legend_handles_labels()
        ax11.legend(handles, labels, loc='upper right',
                    prop={'family': 'Times New Roman' if lang == 'en' else 'SimSun', 'size': 16})

        ax21 = fig.add_subplot(212)
        # ax1.set_ylim(0, 2, 0.2)
        # ax21.set_xlim(0, 13)
        # ax21.set_xticks([])
        ax21.tick_params(axis='x', labelsize=20)  # 设置字体大小为10
        ax21.tick_params(axis='y', labelsize=20)  # 设置字体大小为10

        # ax22 = ax21.twinx()  # 使用twinx()，得到与ax1 对称的ax2,共用一个x轴，y轴对称（坐标不对称）
        if alum:
            ax21.set_ylim(10, 20)
        else:
            ax21.set_ylim(1.4, 2.2)
        # ax22.tick_params(axis='y', labelsize=20)
        ax21.plot(range(len(y_test)), y_test, color_list[2:][0] + '-', label='实测值')
        ax21.plot(range(len(result_grula)), result_grula, color_list[2:][1] + '-',
                  label='预测值')
        ax21.set_xlabel('样本', fontsize=20, fontproperties='SimSun' if lang == 'zh' else 'Times New Roman')
        ax21.set_ylabel('混凝剂用量（mg/L）' if alum else '出水浊度（NTU）', fontsize=20,
                        fontproperties='SimSun' if lang == 'zh' else 'Times New Roman')
        handles, labels = ax21.get_legend_handles_labels()

        ax21.legend(handles, labels, loc='upper right',
                    prop={'family': 'Times New Roman' if lang == 'en' else 'SimSun', 'size': 16})
        plt.savefig(f'pics/数据驱动/{"alum" if alum else "ntu"}_{int(time.time())}.png', dpi=300, bbox_inches='')
        plt.show()
        return y_test, result_grula

    def GRU_show(self):
        col_list = []
        std_list = []
        rmse_list = []
        mape_list = []
        r_list = []
        r2_list = []
        for f in [True, False]:
            y_test, data = self.GRU_LA_show(f)
            col_list.append('混凝剂用量' if f else '出水浊度')
            std_list.append(self.metrics.std(data))
            rmse_list.append(self.metrics.rmse(y_test, data))
            mape_list.append(self.metrics.mape(y_test, data))
            r_list.append(self.metrics.corr(y_test, data))
            r2_list.append(self.metrics.r2(y_test, data))

        pd_data = pd.DataFrame({
            '标准差': std_list,
            'e': std_list,
            '相关系数': r_list,
            '决定系数': r2_list,
            'RMSE': rmse_list,
            'MAPE(%)': mape_list
        }, index=col_list)
        self.logger.info(pd_data)
        pd_data.to_csv(f'data/数据驱动/GRU_SHOW_{int(time.time())}.csv')

    def ano_detect(self):
        df_data_source = self.model_factory.load_data()[0]
        # 重构训练数据，设定阈值，整体重构，根据阈值剔除数据
        self.model_factory.remake(df_data_source, df_data_source)

    def taylor(self):
        data_path = 'data/数据驱动/compare_1711884112.csv'
        data = pd.read_csv(data_path)
        Plot.taylor(0.3, 1.0, 0.1, 0.78, data)

    def model_compare(self):
        df_data, df_target = self.model_factory.load_data()
        error_dict = {}
        predict_dict = {}
        threshold_dict = {ProjectModel.RF.value: 6.5,
                          ProjectModel.GRU.value: 1.5,
                          ProjectModel.LSTM.value: 2.8,
                          ProjectModel.GRU_GA.value: 0.8,
                          ProjectModel.GRU_LA.value: 0.4,
                          ProjectModel.GRU_LA_EWC.value: 0.2}
        # 对比模型列表
        compare_list = [ProjectModel.GRU.value, ProjectModel.GRU_LA.value, ProjectModel.GRU_GA.value,
                        ProjectModel.LSTM.value, ProjectModel.RF.value]
        y_test = 0
        for m in compare_list:
            self.logger.info(m)
            alum_model, error_list, result, y_test, h = self.model_factory.model_train(df_data, df_target, m=m)
            error_dict[m] = error_list
            predict_dict[m] = result

        for items in error_dict:
            error_dict[items] = np.where(abs(error_dict[items]) > threshold_dict[items])

        del_index = list(error_dict[ProjectModel.RF.value][0])
        for items in error_dict:
            del_index += list(error_dict[items][0])
        del_index = np.array(list(set(del_index)))

        y_test = np.delete(y_test, del_index)
        col_list = []
        std_list = []
        rmse_list = []
        mape_list = []
        r_list = []
        r2_list = []
        for items in predict_dict:
            predict_dict[items] = np.delete(predict_dict[items], del_index)
            col_list.append(items)
            data = predict_dict[items]
            std_list.append(self.metrics.std(data))
            rmse_list.append(self.metrics.rmse(y_test, data))
            mape_list.append(self.metrics.mape(y_test, data))
            r_list.append(self.metrics.corr(y_test, data))
            r2_list.append(self.metrics.r2(y_test, data))

        pd_data = pd.DataFrame({
            '标准差': std_list,
            'e': std_list,
            '相关系数': r_list,
            '决定系数': r2_list,
            'RMSE': rmse_list,
            'MAPE(%)': mape_list
        }, index=col_list)
        self.logger.info(pd_data)
        pd_data.to_csv(f'data/数据驱动/compare_{int(time.time())}.csv', index_label='模型')
        Plot.paint_double('混凝剂投加量（mg/L）', lang='zh', smooth=True,
                          data_dict={'实际值': y_test, **predict_dict},
                          to_save=f'pics/数据驱动/compare_{int(time.time())}.png')

    def ewc_test(self):
        df_data, df_target = self.model_factory.load_data()
        self.model_factory.model_train(df_data, df_target, ewc_test=True)


if __name__ == '__main__':
    project = Project()
    # project.knowledge_compare()
    # project.GRU_show()
    # project.model_compare()
    # project.ano_detect()
    project.ewc_test()
