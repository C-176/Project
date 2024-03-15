import datetime
import math
import os
import struct
import time

import numpy
import numpy as np
from matplotlib import pyplot as plt

from alum.public_util import *
import traceback
import snap7


def processData(value_list, ind1):
    # 3(划掉) 2σ去噪
    value_value = [item[ind1] for item in value_list]
    mean = numpy.mean(value_value)
    std = numpy.std(value_value)
    results = []
    for item in value_list:
        if mean + 2 * std >= item[ind1] >= mean - 2 * std:
            results.append(item)
        else:
            tmp = list(item)
            tmp[ind1] = mean
            results.append(tmp)
    return results


def processListData(value_list):
    mean = numpy.mean(value_list)
    std = numpy.std(value_list)
    result = []
    for item in value_list:
        if mean + 2.5 * std >= item >= mean - 2.5 * std:
            result.append(item)
        else:
            result.append(mean)
    return result


def feedforward_filter(index, before_date, before_time, offset=0):
    """
    前馈参数滤波
    :param offset: 时间偏移量
    :param index: 字段index
    :param before_date: 日期字符串
    :param before_time: 时间字符串
    :return: 滤波值
    """
    if offset == 0:
        before_date, before_time = get_now_date_and_time()
    # 获取18条历史信息 根据19条信息 计算9 组最近滤波值（1-10, ... ,9-18）
    sql = f"SELECT * FROM `feedforward_collection`where Date_S <= '{before_date}' and Time_S <= '{before_time}' order by id DESC  limit  19 "  # 1号池的前馈参数查询
    results = query(sql)

    # 如果查到的数据的最后一条也就是这些数据中最老的一条的时间比现在早0.3个小时，直接返回当前值
    if hour_differ_cal(str(results[-1][1]), str(results[-1][2]), before_date, before_time) * 60 > 30:
        return results[0][index]

    # 查询的数据较多 根据索引只提取我们需要的数组 放置在lis数组中
    lis = []
    results = list(results)
    results = processData(results, index)
    result_number = len(results)
    for i in range(result_number, 19):
        results.append(results[-1])

    for i in range(1, len(results)):  # 取出前18组数
        lis.append(results[i][index])

    # 用于放置计算当前滤波值的10组数据
    li = []
    # 数组中第一个放置当前未滤波的值
    NTU_now = results[0][index]
    li.append(NTU_now)
    # 将9组数据滤波，再追加到li数组当中
    for i in range(0, 9):
        li.append(filter(lis[i:i + 10]))
    # standard_variance = std(li)
    # 最后将li 数组进行滤波，直接返回,
    # filter(li) 滤波后的当前浊度，li当前时刻附近10个浊度滤波值，当前时刻加矾量
    return filter(li)


def Alum_mean(index, Alum, pool):
    """
    将最近十个加矾量取平均
    """
    sql = "select * from cal_forecast%d order by id desc limit 5" % pool
    result = query(sql)
    Alum_resent = []
    for i in result:
        if i[index] != None:
            Alum_resent.append(i[index])
    return (sum(Alum_resent) + Alum) / (len(Alum_resent) + 1)


def NTU_filter(NTU_now, ind2, date_str, time_str, ind3):
    """
    返回 当前的浊度滤波值 、当前时刻最近范围内10个滤波值、当前时刻前加矾量、下一次要查询的日期时间
    """

    # 获取18条历史信息 根据18条信息 计算9 组最近滤波值（1-10, ... ,9-18）
    if ind3 == 1:
        sql = "SELECT * FROM `feedback_collection` where Date_S<='" + date_str + "' and Time_S<='" + time_str + "' order by id DESC LIMIT 18"
    else:
        sql = "SELECT * FROM `feedback_collection` order by id DESC LIMIT 18"
    results = query(sql)

    # 如果查到的数据的最后一条也就是这些数据中最老的一条的时间比现在早0.3个小时，直接返回当前值
    if hour_differ_cal(str(results[-1][1]), str(results[-1][2]), date_str, time_str) * 60 > 30:
        return NTU_now, [], results[0][25], results[0][26]

    # 查询的数据较多 根据索引只提取我们需要的数组放置在lis数组中
    lis = []
    results = list(results)
    result_number = len(results)
    # 保证results里面装满18个数据
    for i in range(result_number, 18):
        results.append(results[-1])

    for i in range(0, len(results)):
        lis.append(results[i][ind2])
    # 用于放置计算当前滤波值的10组数据
    li = []
    # 数组中第一个放置当前未滤波的值
    li.append(NTU_now)
    # 将9组数据滤波，再追加到li数组当中
    for i in range(0, 9):
        li.append(filter(lis[i:i + 10]))
    li = processListData(li)
    # 最后将li 数组进行滤波，直接返回,
    # filter(li) 滤波后的当前浊度，li当前时刻附近9个浊度滤波值和一个实时浊度值，当前时刻加矾量
    # recordtime = datetime.timedelta(seconds=(stop_time - start_time) // 1)
    return filter(li), li, results[-1][25], results[-1][26]


def feedback_filter(value, index, date_str, time_str):
    """
    返回 当前的浊度滤波值 、当前时刻最近范围内10个滤波值、当前时刻前加矾量、下一次要查询的日期时间
    """
    sql = f"SELECT * FROM `feedback_collection` where Date_S <= '{date_str}' and Time_S<='{time_str}' order by id DESC LIMIT 18"
    results = query(sql)
    # 如果查到的数据的最后一条也就是这些数据中最老的一条的时间比现在早0.3个小时，直接返回当前值
    if hour_differ_cal(str(results[-1][1]), str(results[-1][2]), date_str, time_str) * 60 > 30:
        return value
    # 查询的数据较多 根据索引只提取我们需要的数组放置在lis数组中
    lis = []
    results = list(results)
    result_number = len(results)
    # 保证results里面装满18个数据
    for i in range(result_number, 18):
        results.append(results[-1])

    for i in range(0, len(results)):
        lis.append(results[i][index])
    # 用于放置计算当前滤波值的10组数据
    # 数组中第一个放置当前未滤波的值
    li = [value]
    # 将9组数据滤波，再追加到li数组当中
    for i in range(0, 9):
        li.append(filter(lis[i:i + 10]))
    li = processListData(li)
    return filter(li)


def adjust_value(column, value, id):
    """
    调整前馈参数某行某列的值
    :param column: 字段名
    :param value: 新值
    :param id: 行id
    :return:
    """
    value = value if type(value) == str else str(value)

    sql = 'update feedforward_collection set %s = %s where id=%d' % (column, value, id)
    query(sql)


def alum_before_data_write_task():
    info = 0
    while info < restart_time:
        try:
            # 查询最后一条数据的日期时间
            sql = "SELECT lake_datetime FROM `lake_pass_dis_per_sec` ORDER BY `id` DESC LIMIT 1"
            result = None
            NTU_list = None
            if query(sql):
                date1, time1 = str(query(sql)[0][0]).split(' ')
                if hour_differ_cal(date1, time1) * 60 >= 5:  # 数据更新不及时
                    if hour_differ_cal(date1, time1) <= 6:  # 数据更新不及时
                        # 查询feedback_collection表中date，time往后的所有数据
                        sql = "select date_s,time_s,jsll1+jsll2+jsll3+jsll4 jsll from feedback_collection WHERE date_s>=\"%s\" AND time_s>=\"%s\"" % (
                            date1, time1)
                        result = query(sql)
                    else:  # 插入最近10个小时的数据
                        sql = "select date_s,time_s,jsll1+jsll2+jsll3+jsll4 jsll  from feedback_collection order by id desc limit 600"
                        result = query(sql)
                        result = [result[i] for i in range(len(result) - 1, -1, -1)]
                else:
                    sql = 'select date_s,time_s,jsll1+jsll2+jsll3+jsll4 jsll  from feedback_collection order by id desc limit 1'
                    result = query(sql)
                    sql = 'select syd_yszd_cj from feedforward_collection order by id desc limit 1'
                    NTU_list = query(sql)[0]
            else:
                sql = "select date_s,time_s,jsll1+jsll2+jsll3+jsll4 jsll  from feedback_collection order by id desc limit 600"
                result = query(sql)
                result = [result[i] for i in range(len(result) - 1, -1, -1)]

            for i in result:
                dis = i[2] / 60 / 3.078
                datetime1 = str(i[0]) + ' ' + str(i[1])
                NTU = 0 if not NTU_list else NTU_list[0]

                sql = 'insert into lake_pass_dis_per_sec (lake_datetime,lake_water_flow,lake_pass_dis,now_NTU) values (\"%s\",\"%f\",\"%f\",\"%f\")' % (
                    datetime1, i[2], dis, NTU)
                query(sql)

            save_record = 60 * 24 * 2
            # 查询出此表中数据个数，如果大于60*24*2，保留最后60*24*2条数据
            sql = 'select count(*),max(id) from lake_pass_dis_per_sec'
            result = query(sql)
            if result[0][0] > save_record:
                sql = 'delete from lake_pass_dis_per_sec where id<=%d' % (result[0][1] - save_record)
                query(sql)
            return
        except Exception as e:
            info += 1
            time.sleep(1)
    if info >= restart_time:
        print('[数据库-前加矾滞后计算]操作失败')
