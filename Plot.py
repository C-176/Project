import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches
from scipy.interpolate import make_interp_spline

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
marker_list = ['.', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D',
               'd']

RADIS = 1.1

tylor_color_list = ['#481b6d', '#3f4788', '#2e6e8e', '#21918c', '#2db27d', '#73d056', '#d0e11c']

Simsun = FontProperties(fname=r"C:\Windows\Fonts\simsunb.ttf")
Times = FontProperties(fname=r"C:\Windows\Fonts\times.ttf")
mpl.rcParams['axes.unicode_minus'] = False


class Plot:

    @staticmethod
    def paint_for_knowledge_driven(dicts, lang='en', title=None, sub_col=None):
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
        if title:
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
        plt.savefig('pics/知识驱动/' + str(int(time.time())) + '.png', dpi=300, bbox_inches='')
        plt.show()
    @staticmethod
    def taylor(r_small, r_big, r_interval, ref_std, ENN, RF, small, big):
        ENN = pd.read_excel('ENN09-21-10-11-59.xlsx', sheet_name='Sheet1')
        RF = pd.read_excel('RF09-21-10-11-24.xlsx', sheet_name='Sheet1')
        fig = plt.figure(figsize=(8, 6), dpi=140)
        axe = plt.subplot(1, 1, 1, projection='polar')
        axe.set_title('Taylor Diagram of Effluent Turbidity Prediction Models', fontproperties=Times, fontsize=18,
                      y=1.08)
        axe.set_thetalim(thetamin=0, thetamax=90)
        # r_small, r_big, r_interval = 0.4, 2 * RADIS, 0.2
        axe.set_rlim(r_small, r_big)
        rad_list = [0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1]  # 需要显示数值的主要R的值
        minor_rad_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.86, 0.87, 0.88, 0.89,
                          0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1]  # 需要显示刻度的次要R的值
        angle_list = np.rad2deg(np.arccos(rad_list))
        angle_list_rad = np.arccos(rad_list)
        angle_minor_list = np.arccos(minor_rad_list)
        axe.set_thetagrids(angle_list, rad_list)

        axe.set_rgrids([])
        labels = axe.get_xticklabels() + axe.get_yticklabels()
        [label.set_fontproperties(FontProperties(fname=r"C:\Windows\Fonts\times.ttf", size=16)) for label in labels]
        axe.grid(False)

        angle_linewidth, angle_length, angle_minor_length = 0.8, 0.02, 0.01
        tick = [axe.get_rmax(), axe.get_rmax() * (1 - angle_length)]
        tick_minor = [axe.get_rmax(), axe.get_rmax() * (1 - angle_minor_length)]
        for t in angle_list_rad:
            axe.plot([t, t], tick, lw=angle_linewidth, color="k")  # 第一个坐标是角度（角度制），第二个是距离
        for t in angle_minor_list:
            axe.plot([t, t], tick_minor, lw=angle_linewidth, color="k")  # 第一个坐标是角度（角度制），第二个是距离

        for i in np.arange(r_small, r_big, r_interval):
            if abs(ref_std - i) >= r_interval:
                axe.text(0, i, s='\n' + str(round(i, 2)), fontproperties=Times, fontsize=18,
                         ha='center', va='top')  # text的第一个坐标是角度（弧度制），第二个是距离
            axe.text(np.pi / 2, i, s=str(round(i, 2)) + '  ', fontproperties=Times, fontsize=18,
                     ha='right', va='center')  # text的第一个坐标是角度（弧度制），第二个是距离
        axe.text(0, ref_std, s='\n' + 'REF', fontproperties=Times, fontsize=18,
                 ha='center', va='top')  # text的第一个坐标是角度（弧度制），第二个是距离
        # axe.set_xlabel('Normalized', fontproperties=Times, labelpad=18, fontsize=18)
        axe.set_ylabel('Standard deviation', fontproperties=Times, labelpad=40, fontsize=18)
        axe.text(np.deg2rad(45), r_big + 0.14, s='COR', fontproperties=Times, fontsize=18, ha='center', va='bottom',
                 rotation=-45)
        # 绘制以REF为原点的圈
        for i in np.arange(r_small, r_big, r_interval):
            circle = plt.Circle((ref_std - r_small, 0), i - r_small, transform=axe.transData._b, facecolor=(0, 0, 0, 0),
                                edgecolor='blue', linestyle='--', linewidth=0.8)
            axe.add_artist(circle)
        axe.plot(0, ref_std, '.', color=tylor_color_list[0], markersize=15)
        # axe.add_artist(circle)
        # 绘制以原点为圆心的圆
        for i in np.arange(r_small, r_big, r_interval):
            circle = plt.Circle((0, 0), i - r_small, transform=axe.transData._b, facecolor=(0, 0, 0, 0),
                                edgecolor='red',
                                linestyle='--')
            axe.add_artist(circle)

        line_list = [0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1]
        for i in line_list:
            axe.plot([0, np.arccos(i)], [r_small, r_big], lw=0.7, color='green', linestyle='--')

        ENN = np.array(ENN)
        RF = np.array(RF)

        def get_color(x, type='ENN', train=False):
            # 将x在enn_test_std中的位置取整，作为颜色的索引
            # 将ENN[9]从小到大排序，取出对应的索引
            if type == 'ENN' and not train:
                val = ENN[x, -2]
            elif type == 'ENN' and train:
                val = ENN[x, 5]
            elif type == 'RF' and not train:
                val = RF[x, -2]
            elif type == 'RF' and train:
                val = RF[x, 5]
            print(x + 1, ":", val)
            index = int(val / big * len(tylor_color_list))
            # # 找到list中与x差值最小的数的位置
            # index = list.index(min(list, key=lambda y: abs(y - x)))
            # # 将index映射到颜色列表中
            # index = index / len(list) * len(tylor_color_list)
            # # 取出x在ENN[9]中的位置
            print(int(index))
            return tylor_color_list[int(index)]

        for (index, (i, j)) in enumerate(zip(ENN, RF)):
            if index > 7:
                continue
            rf_test_std, rf_train_std, rf_cor = j[7], j[8], j[9]
            enn_test_std, enn_train_std, enn_cor = i[7], i[8], i[9]
            axe.plot(float(np.arccos(rf_cor)), rf_test_std, 'D', markersize=10,
                     color=get_color(index, 'RF'), label=f'RF')
            axe.text(np.arccos(rf_cor), rf_test_std, s=f'{index + 1}', fontproperties=Times, fontsize=18)
            axe.plot(np.arccos(enn_cor), enn_test_std, 'o', markersize=10,
                     color=get_color(index), label=f'ENN')
            axe.text(np.arccos(enn_cor), enn_test_std, s=f'{index + 1}', fontproperties=Times, fontsize=18)
        labels = ['ENN', 'RF']
        shape = ['o', 'D']

        # legend标签列表
        # 用label和color列表生成mpatches.Patch对象，它将作为句柄来生成legend

        class HandlerEllipse(HandlerPatch):
            def create_artists(self, legend, orig_handle,
                               xdescent, ydescent, width, height, fontsize, trans):
                print('ydescent', ydescent)
                center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
                p = mpatches.Ellipse(xy=center, width=height + xdescent,
                                     height=height + ydescent)
                self.update_prop(p, orig_handle, legend)
                p.set_transform(trans)
                return [p]

        class HandlerEllipse1(HandlerPatch):

            def create_artists(self, legend, orig_handle,
                               xdescent, ydescent, width, height, fontsize, trans):
                print('xdescent', ydescent)
                center = 0.5 * width, 0
                p = mpatches.Rectangle(xy=center, width=height + xdescent,
                                       height=height + ydescent, angle=45.0)
                self.update_prop(p, orig_handle, legend)
                p.set_transform(trans)
                return [p]

        c = [mpatches.Circle((0.5, 0.5), radius=0.25, edgecolor="black", facecolor='yellow'),
             mpatches.Rectangle((0, 0), 1, 1, 45.0, edgecolor="black", facecolor='yellow')]
        # plt.legend(c, labels, bbox_to_anchor=(1.33, 1.15), loc='upper right',
        #            prop={'family': 'Times New Roman', 'size': 18},
        #            handler_map={mpatches.Circle: HandlerEllipse(), mpatches.Rectangle: HandlerEllipse1()})

        # 添加颜色条
        M = 50
        a = 1
        delta_r = 1 / M
        space_theta = np.radians(np.linspace(0, 0))
        space_r = np.arange(0, a, delta_r)
        T = np.random.uniform(small, big, M * len(space_r)).reshape((M, len(space_r)))

        r, theta = np.meshgrid(space_theta, space_r)

        # contourplot = axe.contourf(r, theta, T)
        # plt.rcParams['font.family'] = 'Times New Roman'
        # plt.rcParams['font.size'] = 15
        #
        # cb = plt.colorbar(contourplot, pad=0.1, extend='both', shrink=0.8)
        # cb.set_label('RMSE (mg/L)')
        plt.show()
    @staticmethod
    def paint_error(data, data1, column: list, legend_str, pause=False, pause_time=3, to_save=False, smooth=False,
                    lang='en',
                    title=None):
        fig, ax = plt.subplots(1, 1)

        fig.set_size_inches(9, 6)
        fig.set_dpi(100)

        # 共享x轴，生成次坐标轴
        ax_sub = ax.twinx()
        # 设置主次y轴的title
        ax.set_ylabel(column[0], fontsize=20, fontproperties='SimSun' if lang == 'zh' else 'Times New Roman')
        ax_sub.set_ylabel(column[1], fontsize=20, fontproperties='SimSun' if lang == 'zh' else 'Times New Roman')
        # 设置x轴title
        ax.set_xlabel('Sample' if lang == 'en' else '样本', fontsize=20,
                      fontproperties='SimSun' if lang == 'zh' else 'Times New Roman')
        # 设置图片title
        # ax.set_title('Effluent turbidity residuals', fontsize=20)
        ax.set_title(title, fontsize=20)
        data = np.array(data)
        data1 = np.array(data1)
        ax.set_ylim(np.min(data) * 0.5 if np.min(data) > 0 else np.min(data) * 1.5, np.max(data) * 1.7)
        ax_sub.set_ylim(np.min(data1) * 0.5 if np.min(data1) > 0 else np.min(data1) * 1.5, np.max(data1) * 1.7)
        ax.tick_params(axis='x', labelsize=20, pad=5)
        ax.tick_params(axis='y', labelsize=20, pad=5)
        ax_sub.tick_params(axis='y', labelsize=20, pad=5)

        plt.grid(linestyle="--", linewidth=1)
        if smooth:
            xy_s = Plot.smooth_xy(range(len(data)), data)
            xy_s1 = Plot.smooth_xy(range(len(data1)), data1)
        else:
            xy_s = (range(len(data)), data)
            xy_s1 = (range(len(data)), data1)

        # 绘图
        # l1, = ax.plot(xy_s[0], xy_s[1], '#440357', label=f'{legend_str}', marker='o', markersize=5, markerfacecolor='white')
        # 设置图例的背景色为白色

        l1, = ax.plot(xy_s[0], xy_s[1], '#36b77b', label=f'{legend_str}', marker='o', markersize=5, markerfacecolor='white')
        l2, = ax_sub.plot(xy_s1[0], xy_s1[1], '#440357', label=f'{legend_str}', marker='*', markersize=8,
                          markerfacecolor='white')
        # 放置图例

        plt.legend(handles=[l1], loc='upper right',
                   prop={'family': 'SimSun' if lang == 'zh' else 'Times New Roman', 'size': 20}, facecolor='white')

        if to_save:
            plt.savefig(fr'{column[0]}.png', bbox_inches='tight')
    @staticmethod
    def smooth_xy(lx, ly):
        """数据平滑处理

        :param lx: x轴数据，数组
        :param ly: y轴数据，数组
        :return: 平滑后的x、y轴数据，数组 [slx, sly]
        """
        x = np.array(lx)
        y = np.array(ly)
        x_smooth = np.linspace(x.min(), x.max(), 600)
        y_smooth = make_interp_spline(x, y)(x_smooth)
        return [x_smooth, y_smooth]
    @staticmethod
    def show_loss(history, pause=False):
        lang = 'zh'
        plt.figure(figsize=(8, 6), dpi=100)  # 设置画布大小，像素
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        data1 = history.history['loss']
        data2 = history.history['val_loss']
        label1 = '训练集'
        label2 = '验证集'
        data1, data2 = np.array(data1), np.array(data2)
        # plt.ylim((np.min(data1) * 0.5 if np.min(data1) > 0 else np.min(data1) * 1.5, np.max(data1) * 1.5))
        plt.ylim((0, 0.03))
        plt.xlabel('Sample' if lang == 'en' else '迭代次数', fontsize=20,
                   fontproperties='Times New Roman' if lang == 'en' else 'SimSun')
        plt.ylabel('损失（MSE）', fontsize=20, fontproperties='SimSun' if lang == 'zh' else 'Times New Roman')

        xy_s1 = (range(len(data2)), data2)
        xy_s2 = (range(len(data1)), data1)
        plt.plot(xy_s1[0], xy_s1[1], '#440357', label=label2)
        plt.scatter(xy_s2[0], xy_s2[1], color='#36b77b', label=label1, s=14)

        plt.legend(loc='upper right', facecolor='#fff',
                   prop={'family': 'Times New Roman' if lang == 'en' else 'SimSun', 'size': 20})

        # plt.savefig(f'图/loss.png', bbox_inches='tight')
        plt.pause(3)
        plt.close()
    @staticmethod
    # 双 线图
    def paint_double(column, data1, label1, data2, label2, to_save=False, show=True, smooth=False, lang='en',
                     sub_ax=None):
        plt.figure(figsize=(19, 10), dpi=100)  # 设置画布大小，像素
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        data1, data2 = np.array(data1), np.array(data2)
        # plt.ylim((np.min(data2) * 0.7 if np.min(data2) > 0 else np.min(data2) * 1.3, np.max(data2) * 1.3))

        plt.xlabel('Sample' if lang == 'en' else '样本', fontsize=20,
                   fontproperties='Times New Roman' if lang == 'en' else 'SimSun')
        plt.ylabel(column, fontsize=20, fontproperties='SimSun' if lang == 'zh' else 'Times New Roman')

        if smooth:
            xy_s1 = Plot.smooth_xy(range(len(data2)), data2)
            xy_s2 = Plot.smooth_xy(range(len(data1)), data1)
        else:
            xy_s1 = (range(len(data2)), data2)
            xy_s2 = (range(len(data1)), data1)

        plt.plot(xy_s1[0], xy_s1[1], '#440357', label=label2)
        plt.plot(xy_s2[0], xy_s2[1], '#36b77b', label=label1)
        # plt.scatter(xy_s2[0] if sub_ax is None else sub_ax, xy_s2[1], color='#36b77b', label=label1, s=8)

        plt.legend(loc='upper right', facecolor='#fff',
                   prop={'family': 'Times New Roman' if lang == 'en' else 'SimSun', 'size': 20})

        if to_save:
            plt.savefig(f'{to_save}', format='svg', bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

        # pass


if __name__ == '__main__':
    plot = Plot()
    # 浊度
    # taylor(0.19, 0.3, 0.02, ENN.iloc[11, 8], ENN, RF, 0 , 0.3)
    # 矾
    # plot.taylor(0.4, 2.2, 0.3, ENN.iloc[0, 8], ENN, RF, 0, 2.8)
    # axe = plt.subplot(1, 2, 2, projection='polar')
    # axe.plt.show()
