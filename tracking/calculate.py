import pandas as pd
import numpy as np
import os
from scipy.signal import butter, filtfilt

# 设计滤波器
def bandpass_filter(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band', analog=False)
    return b, a
# 应用滤波器
def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    b, a = bandpass_filter(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y
def calculate(file_path , size, fps):
    dt = 1 / float(fps)
    lowcut = 0.05
    highcut = (1/dt)/2 - (1/dt)/10
    # data = []
    with open(file_path, 'r') as txt_file:  # 读取文本文件
        lines = txt_file.readlines()
    # for line in lines:
    #     if line.strip() != '':
    #         str = line.strip().rstrip(',').split(',')
    #         data.extend([list(map(float, str))])
    # 处理数据，创建DataFrame
    data = [list(map(float, line.strip().split('\t'))) for line in lines]
    # data = [list(map(float, line.strip().rstrip(',').split(','))) for line in lines]
    df = pd.DataFrame(data, columns=['X', 'Y', 'Width', 'Height'])

    # 毫米转换为米
    df['X'] = df['X'] / 1000
    df['Y'] = df['Y'] / 1000
    # 计算位移，第一行留空
    df['X位移'] = (df['X'] - df['X'].shift(1)) * size
    df['Y位移'] = (df['Y'] - df['Y'].shift(1)) * size
    df['X相对位移'] = df['X'] - df['X'].iloc[0]
    df['Y相对位移'] = df['Y'] - df['Y'].iloc[0]
    MAX_X = df['X相对位移'].abs().max()
    MAX_Y = df['Y相对位移'].abs().max()

    # 计算速度
    df['X速度'] = (df['X'] - df['X'].shift(1)) * size / dt
    df['Y速度'] = (df['Y'] - df['Y'].shift(1)) * size / dt
    # 应用滤波
    signal_XV = df['X速度'].iloc[2:].values
    signal_YV = df['Y速度'].iloc[2:].values
    filtered_signal_XV = butter_bandpass_filter(signal_XV, lowcut, highcut, 1/dt)
    filtered_signal_YV = butter_bandpass_filter(signal_YV, lowcut, highcut, 1/dt)
    df['X速度（滤波）'] = pd.DataFrame(filtered_signal_XV)
    df['Y速度（滤波）'] = pd.DataFrame(filtered_signal_YV)

    # # 计算合成速度
    # df['合成速度'] = np.sqrt(df['X速度'] ** 2 + df['Y速度'] ** 2)

    # 计算加速度，前两行留空
    df['X加速度'] = (df['X'] - 2 * df['X'].shift(1) + df['X'].shift(2)) * size / dt ** 2
    df['Y加速度'] = (df['Y'] - 2 * df['Y'].shift(1) + df['Y'].shift(2)) * size / dt ** 2
    # 应用滤波
    signal_XA = df['X加速度'].iloc[2:].values
    signal_YA = df['Y加速度'].iloc[2:].values
    filtered_signal_XA = butter_bandpass_filter(signal_XA, lowcut, highcut, 1/dt)
    filtered_signal_YA = butter_bandpass_filter(signal_YA, lowcut, highcut, 1/dt)
    df['X加速度（滤波）'] = pd.DataFrame(filtered_signal_XA)
    df['Y加速度（滤波）'] = pd.DataFrame(filtered_signal_YA)

    # # 计算合成加速度
    # df['合成加速度'] = np.sqrt(df['X加速度'] ** 2 + df['Y加速度'] ** 2)
    # # df['合成加速度'] = math.sqrt(math.pow(df['X加速度'], 2) + math.pow(df['Y加速度'], 2))

    # PGA = df['合成加速度'].max() / 1000
    # PGV = df['合成速度'].max() / 1000
    max_XV = abs((df['X速度（滤波）'])).max()
    max_YV = abs((df['Y速度（滤波）'])).max()
    max_XA = abs((df['X加速度（滤波）'])).max() / 10
    max_YA = abs((df['Y加速度（滤波）'])).max() / 10
    # max_XV = abs((df['X速度（滤波）'])).max()
    # max_YV = abs((df['Y速度（滤波）'])).max()
    # max_XA = abs((df['X加速度（滤波）'])).max()
    # max_YA = abs((df['Y加速度（滤波）'])).max()
    # PGA = np.sqrt(abs((df['X加速度（滤波）'])).max() ** 2 + abs((df['Y加速度（滤波）'])).max() ** 2) / 100000
    # PGV = np.sqrt(abs((df['X速度（滤波）'])).max() ** 2 + abs((df['Y速度（滤波）'])).max() ** 2) / 100000
    PGV = np.sqrt(max_XV * max_XV + max_YV * max_YV)
    PGA = np.sqrt(max_XA * max_XA + max_YA * max_YA)

    Ia = 3.17 * np.log10(PGA) + 6.59
    Iv = 3.00 * np.log10(PGV) + 9.77

    if Ia >= 6.0 and Iv >= 6.0:
        Ii = Iv
    elif Ia < 6.0 or Iv < 6.0:
        Ii = (Ia + Iv) / 2

    if Ii < 1.0:
        Ii = 1.0

    if Ii > 12.0:
        Ii = 12.0

    # Ii = round(Ii, 1)
    #
    # if 1.0 <= Ii < 1.5:
    #     earthquake_intensity = 1
    # elif 1.5 <= Ii < 2.5:
    #     earthquake_intensity = 2
    # elif 2.5 <= Ii < 3.5:
    #     earthquake_intensity = 3
    # elif 3.5 <= Ii < 4.5:
    #     earthquake_intensity = 4
    # elif 4.5 <= Ii < 5.5:
    #     earthquake_intensity = 5
    # elif 5.5 <= Ii < 6.5:
    #     earthquake_intensity = 6
    # elif 6.5 <= Ii < 7.5:
    #     earthquake_intensity = 7
    # elif 7.5 <= Ii < 8.5:
    #     earthquake_intensity = 8
    # elif 8.5 <= Ii < 9.5:
    #     earthquake_intensity = 9
    # elif 9.5 <= Ii < 10.5:
    #     earthquake_intensity = 10
    # elif 10.5 <= Ii < 11.5:
    #     earthquake_intensity = 11
    # elif 11.5 <= Ii < 12.0:
    #     earthquake_intensity = 12
    return round(Ii, 1), PGA, PGV, MAX_X, MAX_Y
    # return Ii, earthquake_intensity, PGA, PGV, MAX_X, MAX_Y