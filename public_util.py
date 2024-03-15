import datetime
import math
import random

import numpy as np
import pymysql

from dbutils.pooled_db import PooledDB
import snap7
import struct
import time
import traceback

from matplotlib import pyplot as plt

restart_time = 9

# 坐标轴的刻度设置向内(in)或向外(out)
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置中文为宋体
# plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 用来正常显示中文标签
plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']  # 设置英文为Times New Roman
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['axes.linewidth'] = 2  # 用来正常显示负号
plt.rcParams['xtick.major.width'] = 3  # 用来正常显示负号
plt.rcParams['xtick.major.size'] = 3  # 用来正常显示负号
plt.rcParams['xtick.major.pad'] = 3  # 用来正常显示负号
plt.rcParams['ytick.major.width'] = 3  # 用来正常显示负号
plt.rcParams['ytick.major.size'] = 3  # 用来正常显示负号
plt.rcParams['ytick.major.pad'] = 1  # 用来正常显示负号

color_list = ['b', 'g', 'r', 'y', 'k', 'p']
line_style = ['o', 'd', '*', '>']


def get_now_date_and_time():
    """
    获取当前的日期和时间
    :return: 日期，时间
    """
    nowDate = time.strftime('%Y-%m-%d', time.localtime())
    nowTime = time.strftime('%H:%M:%S', time.localtime())
    return nowDate, nowTime


def std(records):
    """
    计算标准差
    :param records: 列表
    :return: 标准差
    """
    average = sum(records) / len(records)
    variance = sum([(x - average) ** 2 for x in records]) / len(records)
    return math.sqrt(variance)


def check_limit_setting(alum_tjl, sx, xx):
    if alum_tjl < xx:
        alum_tjl = xx
    elif alum_tjl > sx:
        alum_tjl = sx
    return alum_tjl


def is_sustained_high(data, threshold, period):
    if len(data) < period:
        return False
    else:
        # 如果有80%大于threshold,就算持续超出阈值，则返回True
        # 否则返回False
        count = 0
        for i in range(0, len(data)):
            # 求i到i+int(len(data)/period)之间的数据的均值
            if data[i] > threshold:
                count += 1
        return count / period > 0.8


def is_sustained_down(data, threshold, period):
    if len(data) < period:
        return False
    else:
        # 如果有80%小于于threshold,就算持续低于阈值，则返回True
        # 否则返回False
        count = 0
        for i in range(0, len(data)):
            if data[i] < threshold:
                count += 1
        return count / period > 0.8


def double_lable_paint_test(dicts, lang='en', title=None, sub_col=None):
    fig = plt.figure(figsize=(12, 9), dpi=100)  # 设置画布大小，像素
    fig.subplots_adjust(top=0.85)
    # ax1显示y1  ,ax2显示y2
    ax11 = fig.add_subplot(311)
    ax11.set_ylim(10, 200)
    # ax1.set_xlim(0, 13)
    ax11.tick_params(axis='x', labelsize=20)  # 设置字体大小为10
    ax11.tick_params(axis='y', labelsize=20)  # 设置字体大小为10
    ax11.set_xticks([])
    ax12 = ax11.twinx()  # 使用twinx()，得到与ax1 对称的ax2,共用一个x轴，y轴对称（坐标不对称）
    ax12.set_ylim(0, 60)
    ax12.tick_params(axis='y', labelsize=20)
    for index, (key, value) in enumerate(dicts['变异系数'][0].items()):
        ax11.plot(range(1, len(value) + 1), value, color_list[:2][index] + '-', marker=line_style[:][index],
                  label=key)
    for index, (key, value) in enumerate(dicts['变异系数'][1].items()):
        ax12.plot(range(1, len(value) + 1), value, color_list[2:][index] + '-.', marker=line_style[:][index],
                  label=key)
    plt.title(title, fontdict={'fontsize': 20, 'fontweight': 'bold',
                               'fontfamily': 'SimSun' if lang == 'zh' else 'Times New Roman'})

    ax11.set_ylabel('变异系数（%）', fontsize=20, fontproperties='SimSun' if lang == 'zh' else 'Times New Roman')
    if sub_col:
        ax12.set_ylabel(sub_col, fontsize=20, fontproperties='SimSun' if lang == 'zh' else 'Times New Roman')
    # 获取图例对象
    handles, labels = ax11.get_legend_handles_labels()
    handles2, labels2 = ax12.get_legend_handles_labels()

    # 合并图例
    # ax11.legend(handles + handles2, labels2 + labels, loc='upper right',
    #            prop={'family': 'Times New Roman' if lang == 'en' else 'SimSun', 'size': 16})

    ax21 = fig.add_subplot(312)
    # ax1.set_ylim(0, 2, 0.2)
    # ax21.set_xlim(0, 13)
    ax21.set_xticks([])
    ax21.tick_params(axis='x', labelsize=20)  # 设置字体大小为10
    ax21.tick_params(axis='y', labelsize=20)  # 设置字体大小为10

    # ax22 = ax21.twinx()  # 使用twinx()，得到与ax1 对称的ax2,共用一个x轴，y轴对称（坐标不对称）
    # # ax2.set_ylim(0.1, 1)
    # ax22.tick_params(axis='y', labelsize=20)
    for index, (key, value) in enumerate(dicts['超标数据量'][0].items()):
        ax21.plot(range(1, len(value) + 1), value, color_list[4:][index] + '-', marker=line_style[:][index],
                  label=key)
    # for index, (key, value) in enumerate(dicts['标准差'][1].items()):
    #     ax22.plot(range(1, len(value) + 1), value, color_list[2:][index] + ':', marker=line_style[:][index],
    #               label=key)
    # plt.title(title, fontdict={'fontsize': 20, 'fontweight': 'bold',
    #                            'fontfamily': 'SimSun' if lang == 'zh' else 'Times New Roman'})

    # ax21.set_xlabel('月份', fontsize=20, fontproperties='SimSun' if lang == 'zh' else 'Times New Roman')
    ax21.set_ylabel('超标数据量', fontsize=20, fontproperties='SimSun' if lang == 'zh' else 'Times New Roman')
    # if sub_col:
    #     ax22.set_ylabel(sub_col, fontsize=20, fontproperties='SimSun' if lang == 'zh' else 'Times New Roman')
    # 获取图例对象
    handles, labels = ax21.get_legend_handles_labels()
    # handles2, labels2 = ax22.get_legend_handles_labels()

    # 合并图例
    ax21.legend(handles, labels, loc='upper left',
                prop={'family': 'Times New Roman' if lang == 'en' else 'SimSun', 'size': 16})

    ax31 = fig.add_subplot(313)
    ax31.set_ylim(0, 100)
    # ax1.set_xlim(0, 13)
    ax31.tick_params(axis='x', labelsize=20)  # 设置字体大小为10
    ax31.tick_params(axis='y', labelsize=20)  # 设置字体大小为10

    ax32 = ax31.twinx()  # 使用twinx()，得到与ax1 对称的ax2,共用一个x轴，y轴对称（坐标不对称）
    ax32.set_ylim(0, 2.5)
    ax32.tick_params(axis='y', labelsize=20)
    for index, (key, value) in enumerate(dicts['平均值'][0].items()):
        ax31.plot(range(1, len(value) + 1), value, color_list[:2][index] + '-', marker=line_style[:][index],
                  label=key)
    for index, (key, value) in enumerate(dicts['平均值'][1].items()):
        ax32.plot(range(1, len(value) + 1), value, color_list[2:][index] + '-.', marker=line_style[:][index],
                  label=key)
    ax32.plot(range(1, 13), [1.5] * 12, color='gray', label='出水浊度目标值', linewidth=1)
    # plt.title(title, fontdict={'fontsize': 20, 'fontweight': 'bold',
    #                            'fontfamily': 'SimSun' if lang == 'zh' else 'Times New Roman'})

    ax31.set_xlabel('月份', fontsize=20, fontproperties='SimSun' if lang == 'zh' else 'Times New Roman')
    ax31.set_ylabel('平均值（mg/L）', fontsize=20, fontproperties='SimSun' if lang == 'zh' else 'Times New Roman')
    ax32.set_ylabel('平均值（NTU）', fontsize=20, fontproperties='SimSun' if lang == 'zh' else 'Times New Roman')
    # 获取图例对象
    handles, labels = ax31.get_legend_handles_labels()
    handles2, labels2 = ax32.get_legend_handles_labels()

    # 合并图例
    fig.legend(handles + handles2, labels + labels2, loc='upper center', ncol=3,
               prop={'family': 'Times New Roman' if lang == 'en' else 'SimSun', 'size': 16})
    # ax1.legend(loc='upper left', facecolor='#fff',
    #            prop={'family': 'Times New Roman' if lang == 'en' else 'SimSun', 'size': 16})
    plt.savefig('pics/知识驱动/' + str(time.time()) + '.png', dpi=300, bbox_inches='')
    plt.show()


def double_lable_paint(column, dict1, dict2, lang='en', title=None, sub_col=None):
    fig = plt.figure(figsize=(8, 6), dpi=100)  # 设置画布大小，像素

    # ax1显示y1  ,ax2显示y2
    ax1 = fig.subplots()
    # ax1.set_ylim(0, 2, 0.2)
    # ax1.set_xlim(0, 13)
    ax1.tick_params(axis='x', labelsize=20)  # 设置字体大小为10
    ax1.tick_params(axis='y', labelsize=20)  # 设置字体大小为10

    ax2 = ax1.twinx()  # 使用twinx()，得到与ax1 对称的ax2,共用一个x轴，y轴对称（坐标不对称）
    # ax2.set_ylim(0.1, 1)
    ax2.tick_params(axis='y', labelsize=20)
    for index, (key, value) in enumerate(dict1.items()):
        ax1.plot(range(1, len(value) + 1), value, color_list[:2][index] + '-', marker=line_style[:][index],
                 label=key)
    for index, (key, value) in enumerate(dict2.items()):
        ax2.plot(range(1, len(value) + 1), value, color_list[2:][index] + ':', marker=line_style[:][index],
                 label=key)
    # # 设置背景网格线为虚线
    plt.rcParams['axes.grid'] = True  # 开启网格
    plt.rcParams['grid.color'] = '#CCCCCC'  # 设置网格颜色为灰色
    plt.rcParams['grid.linestyle'] = '--'  # 设置网格为虚线
    plt.title(title, fontdict={'fontsize': 20, 'fontweight': 'bold',
                               'fontfamily': 'SimSun' if lang == 'zh' else 'Times New Roman'})
    # # 设置背景颜色
    plt.gca().set_facecolor('#fff')
    # 设置四周边框颜色宽度
    plt.gca().spines['bottom'].set_color('#000')
    plt.gca().spines['left'].set_color('#000')
    plt.gca().spines['right'].set_color('#000')
    plt.gca().spines['top'].set_color('#000')
    ax1.set_xlabel('月份', fontsize=20, fontproperties='SimSun' if lang == 'zh' else 'Times New Roman')
    ax1.set_ylabel(column, fontsize=20, fontproperties='SimSun' if lang == 'zh' else 'Times New Roman')
    if sub_col:
        ax2.set_ylabel(sub_col, fontsize=20, fontproperties='SimSun' if lang == 'zh' else 'Times New Roman')
    # 获取图例对象
    handles, labels = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    # 合并图例
    ax1.legend(handles + handles2, labels2 + labels, loc='upper right',
               prop={'family': 'Times New Roman' if lang == 'en' else 'SimSun', 'size': 16})
    # ax1.legend(loc='upper left', facecolor='#fff',
    #            prop={'family': 'Times New Roman' if lang == 'en' else 'SimSun', 'size': 16})
    plt.savefig('pics/知识驱动/' + str(time.time()) + '.png', dpi=300, bbox_inches='')
    plt.show()


# 双 线图
def paint_double(column, data1, label1, data2, label2, to_save=False, show=True, smooth=False, lang='en', sub_ax=None):
    plt.figure(figsize=(8, 6), dpi=100)  # 设置画布大小，像素
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    data1, data2 = np.array(data1), np.array(data2)
    # plt.ylim((np.min(data1) * 0.5 if np.min(data1) > 0 else np.min(data1) * 1.5, np.max(data1) * 1.5))

    plt.xlabel('Sample' if lang == 'en' else '样本', fontsize=20,
               fontproperties='Times New Roman' if lang == 'en' else 'SimSun')
    plt.ylabel(column, fontsize=20, fontproperties='SimSun' if lang == 'zh' else 'Times New Roman')

    # xy_s = smooth_xy(range(len(data2)), data2)

    # plt.plot(xy_s[0], xy_s[1], '#2e617d', label=label2)
    xy_s1 = (range(len(data2)), data2)
    xy_s2 = (range(len(data1)), data1)
    # plt.plot(xy_s1[0], xy_s1[1], '#440357', label=label2)
    # plt.scatter(xy_s2[0] if sub_ax is None else sub_ax, xy_s2[1], color='#36b77b', label=label1, s=14)

    # 黑白
    plt.plot(xy_s1[0], xy_s1[1], 'black', label=label2)
    plt.plot(xy_s2[0], xy_s2[1], color='black', label=label1, linestyle='--')
    # 画点图，要求点为圈圈

    # plt.scatter(xy_s2[0] if sub_ax is None else sub_ax, xy_s2[1],
    #             label=label1, marker='o', edgecolor='black',
    #             facecolor='white', s=14)

    # plt.plot(xy_s2[0], xy_s2[1], color='w', label=label1, marker='o',markerfacecolor='#36b77b')
    # print([1.05 * i for i in data2])
    if isinstance(xy_s1, np.ndarray):
        list1 = [1.05 * i[0] for i in xy_s1[1]]
        list2 = [0.95 * i[0] for i in xy_s1[1]]
    else:
        list1 = [1.05 * i for i in xy_s1[1]]
        list2 = [0.95 * i for i in xy_s1[1]]
    # plt.fill_between(xy_s1[0], list1, list2,  # 上限，下限
    #                  facecolor='green',  # 填充颜色
    #                  edgecolor='red',  # 边界颜色
    #                  alpha=0.3)  # 透明度
    # 设置背景网格线为虚线
    plt.rcParams['axes.grid'] = True  # 开启网格
    plt.rcParams['grid.color'] = '#CCCCCC'  # 设置网格颜色为灰色
    plt.rcParams['grid.linestyle'] = '--'  # 设置网格为虚线
    #
    # # 设置背景颜色
    plt.gca().set_facecolor('#fff')
    # 设置四周边框颜色宽度
    plt.gca().spines['bottom'].set_color('#000')
    plt.gca().spines['left'].set_color('#000')
    plt.gca().spines['right'].set_color('#000')
    plt.gca().spines['top'].set_color('#000')

    plt.legend(loc='upper right', facecolor='#fff',
               prop={'family': 'Times New Roman' if lang == 'en' else 'SimSun', 'size': 20})

    if to_save:
        plt.savefig(f'{to_save}', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.pause(3)
        plt.close()

        plt.savefig(to_save, bbox_inches='tight')

    plt.show()


def filter(list):
    """
    滤波
    :param list: 滤波列表
    :return: 滤波值
    """
    a = 1 / 55
    cs = []
    for i in range(10):
        cs.append((10 - i) * a)
    res = 0
    for i in range(0, len(cs)):
        res += list[i] * cs[i]
    return res


def linear_filter(list):
    """
    坡度更陡的滤波
    :param list: 长度为10的列表
    :return: 滤波值
    """
    a = 2 / 55
    cs = []
    for i in range(5):
        cs.append((10 - 2.251 * i) * a)
    res = 0
    for i in range(0, len(cs)):
        res += list[i] * cs[i]
    return res


def get_datetime(day_s, time_s, hours_s):
    """
    获取几小时前/后的日期与时间
    :param day_s: 日期
    :param time_s: 时间
    :param hours_s: 小时差
    :return: 日期，时间
    """
    date1 = day_s + " " + time_s
    now = datetime.datetime.strptime(date1, "%Y-%m-%d %H:%M:%S")
    get_days = now + datetime.timedelta(hours=hours_s)
    # 把datetime转成字符串
    day_s = get_days.strftime("%Y-%m-%d")
    time_s = get_days.strftime("%H:%M:%S")
    # print("getDateTime Running--" + day_s+" "+time_s)
    return day_s, time_s


def hour_differ_cal(data_s, time_s, cmp_date='now', cmp_time='now'):
    """
    计算指定时间与 当前时间 的小时差
    :param data_s: 日期
    :param time_s: 时间
    :param cmp_date: 当前日期（默认now）
    :param cmp_time: 当前时间（默认now）
    :return: 小时数
    """
    if (cmp_date == 'now' and cmp_time == 'now'):
        nowdata = time.strftime('%Y-%m-%d', time.localtime(time.time()))
        nowtime = time.strftime('%H:%M:%S', time.localtime(time.time()))
    else:
        nowdata = cmp_date
        nowtime = cmp_time

    nowd2t = datetime.datetime.strptime(nowdata + " " + nowtime, '%Y-%m-%d %H:%M:%S')
    cund2t = datetime.datetime.strptime(data_s + " " + time_s, '%Y-%m-%d %H:%M:%S')
    delta = nowd2t - cund2t
    chours = delta.seconds / 60 / 60 + delta.days * 24
    return chours


def read_plc(ip, dd, db, amount=4):
    client = snap7.client.Client()
    info = 0
    ip3 = ['198.186.202.72', '198.186.202.73', '198.186.202.74', '198.186.202.75', '198.186.202.76']
    ip2 = ['198.186.202.68', '198.186.202.81']
    if ip in ip3:
        slot = 3
    if ip in ip2:
        slot = 2
    client.connect(ip, 0, slot)
    area = snap7.types.Areas.DB
    while info < restart_time:
        try:
            # 读取泵流量
            db_data = client.read_area(area, dd, db, amount)
            data = struct.unpack('!f' if amount == 4 else '!H', db_data)[0]
            info = restart_time + 9
            return data
        except Exception:
            print(traceback.format_exc())
            time.sleep(0.5)
            info += 1
        # finally:
        #     client.disconnect()


def write_plc(ip, dd, db, content, amount=4):
    client = snap7.client.Client()

    info = 0
    ip3 = ['198.186.202.72', '198.186.202.73', '198.186.202.74', '198.186.202.75', '198.186.202.76']
    ip2 = ['198.186.202.68', '198.186.202.81']
    if ip in ip3:
        slot = 3
    if ip in ip2:
        slot = 2
    client.connect(ip, 0, slot)
    area = snap7.snap7types.areas.DB
    content = struct.pack('!H' if amount == 2 else '!f', content)
    while info < restart_time:
        try:
            client.write_area(area, dd, db, content)
            info = restart_time + 9
        except Exception:
            print(traceback.format_exc())
            time.sleep(0.5)
            info += 1
        # finally:
        #     client.disconnect()


def query(sql):
    pool = PooledDB(pymysql, 5, host='localhost', user='root', passwd='chenle123', db='rglr', port=3306, setsession=[
        'SET AUTOCOMMIT = 1'])  # 5为连接池里的最少连接数，setsession=['SET AUTOCOMMIT = 1']是用来设置线程池是否打开自动更新的配置，0为False，1为True
    conn = pool.connection()
    cur = conn.cursor()
    try:
        cur.execute(sql)
        results = cur.fetchall()
    finally:
        cur.close()
        conn.close()
    return results
