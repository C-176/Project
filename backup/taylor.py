import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
import numpy as np

import pandas as pd


RADIS = 1.1
marker_list = ['.', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D',
               'd']
color_list = ['#481b6d', '#3f4788', '#2e6e8e', '#21918c', '#2db27d', '#73d056', '#d0e11c']
ENN = pd.read_excel('ENN09-21-10-11-59.xlsx', sheet_name='Sheet1')
RF = pd.read_excel('RF09-21-10-11-24.xlsx', sheet_name='Sheet1')
print(ENN)
Simsun = FontProperties(fname=r"C:\Windows\Fonts\simsunb.ttf")
Times = FontProperties(fname=r"C:\Windows\Fonts\times.ttf")
mpl.rcParams['axes.unicode_minus'] = False


def taylor(r_small, r_big, r_interval, ref_std, ENN, RF, small, big):
    fig = plt.figure(figsize=(8, 6), dpi=140)
    axe = plt.subplot(1, 1, 1, projection='polar')
    axe.set_title('Taylor Diagram of Effluent Turbidity Prediction Models', fontproperties=Times, fontsize=18, y=1.08)
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
    axe.plot(0, ref_std, '.', color=color_list[0], markersize=15)
    # axe.add_artist(circle)
    # 绘制以原点为圆心的圆
    for i in np.arange(r_small, r_big, r_interval):
        circle = plt.Circle((0, 0), i - r_small, transform=axe.transData._b, facecolor=(0, 0, 0, 0), edgecolor='red',
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
        index = int(val / big * len(color_list))
        # # 找到list中与x差值最小的数的位置
        # index = list.index(min(list, key=lambda y: abs(y - x)))
        # # 将index映射到颜色列表中
        # index = index / len(list) * len(color_list)
        # # 取出x在ENN[9]中的位置
        print(int(index))
        return color_list[int(index)]

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
    from matplotlib.legend_handler import HandlerPatch
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

    import matplotlib.patches as mpatches
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


# 浊度
# taylor(0.19, 0.3, 0.02, ENN.iloc[11, 8], ENN, RF, 0 , 0.3)
# 矾
taylor(0.4, 2.2, 0.3, ENN.iloc[0, 8], ENN, RF, 0, 2.8)
# axe = plt.subplot(1, 2, 2, projection='polar')
# axe.plt.show()
