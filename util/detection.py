# -*- coding:UTF-8 -*-
# author:Lucifer_Chen（zhangchen）
# contact: 17888808985@163.com
# datetime:2024/7/3 9:13

"""
文件说明：  
"""
import argparse
import os
import time
from collections import deque
from glob import glob
from multiprocessing import Process
import numpy as np
import pandas as pd
import pyabf
from lmfit import Parameters, Minimizer
from scipy.optimize import curve_fit
from astropy.modeling import models, fitting

class adept2State:
    def __init__(self, globtime, data, mean, sd):
        """
            Initialize the single step analysis class.
        """
        # initialize the object's metadata (to -1) as class attributes
        self.globtime = globtime
        self.mdOpenChCurrent = -1
        self.mdBlockedCurrent = -1
        self.mdBlockDepth = -1
        self.mdResTime = -1
        self.mdEventStart = -1
        self.mdEventEnd = -1
        self.sd = sd
        self.baseMean = mean
        self.Fs = 1e5
        self.eventData = data
        self.dataPolarity = float(np.sign(np.mean(self.eventData)))
        self.mdBlockDepth = -1
        self.mdResTime = -1
        self.FitTol = 1.e-7
        self.FitIters = 50000
        self.flag = -2

    def __objfunc(self, params, t, data):
        """ single step response model parameters """
        try:
            tau1 = params['tau1'].value
            tau2 = params['tau2'].value
            mu1 = params['mu1'].value
            mu2 = params['mu2'].value
            a = params['a'].value
            b = params['b'].value
            model = stepResponseFunc(t, tau1, tau2, mu1, mu2, a, b)
            return model - data
        except KeyboardInterrupt:
            raise

    def __eventEndIndex(self, dat, mu, sigma):
        try:
            return ([d for d in dat if d[0] < (mu - 2 * sigma)][-1][1] + 1)
        except IndexError:
            return -1

    def __threadList(self, l1, l2):
        """ thread two lists	"""
        try:
            return list(map(lambda x, y: (x, y), l1, l2))
        except KeyboardInterrupt:
            raise

    def __eventStartIndex(self, dat, mu, sigma):
        try:
            return ([d for d in dat if d[0] < (mu - 2.75 * sigma)][0][1] + 1)
        except IndexError:
            return -1

    def FitEvent(self):
        try:
            varyBlockedCurrent = True

            i0 = np.abs(self.baseMean)
            dt = 1000. / self.Fs  # time-step in ms.
            # edat=np.asarray( np.abs(self.eventData),  dtype='float64' )
            edat = self.dataPolarity * np.asarray(self.eventData, dtype='float64')

            blockedCurrent = min(edat)
            tauVal = dt

            estart = self.__eventStartIndex(self.__threadList(edat, list(range(0, len(edat)))), i0, self.sd) - 1
            eend = self.__eventEndIndex(self.__threadList(edat, list(range(0, len(edat)))), i0, self.sd) - 2

            # For long events, fix the blocked current to speed up the fit
            # if (eend-estart) > 1000:
            #	blockedCurrent=np.mean(edat[estart+50:eend-50])

            # control numpy error reporting
            np.seterr(invalid='ignore', over='ignore', under='ignore')

            ts = np.array([t * dt for t in range(0, len(edat))], dtype='float64')
            # pl.plot(ts,edat)
            # pl.show()

            params = Parameters()

            # print self.absDataStartIndex

            params.add('mu1', value=estart * dt)
            params.add('mu2', value=eend * dt)
            params.add('a', value=(i0 - blockedCurrent), vary=varyBlockedCurrent)
            params.add('b', value=i0)
            params.add('tau1', value=tauVal)
            params.add('tau2', value=tauVal, expr='tau1')

            optfit = Minimizer(self.__objfunc, params, fcn_args=(ts, edat,))
            optfit.prepare_fit()

            result = optfit.leastsq(xtol=self.FitTol, ftol=self.FitTol, max_nfev=self.FitIters)

            if result.success:
                if result.params['mu1'].value < 0.0 or result.params['mu2'].value < 0.0:
                    self.flag = -1
                    self.mdBlockDepth = -1
                    self.mdResTime = -1
                    print('invaild')

                else:
                    self.flag = 1
                    self.mdOpenChCurrent = result.params['b'].value
                    # self.mdOpenChCurrent = i0
                    self.mdBlockedCurrent = self.mdOpenChCurrent - result.params['a'].value
                    self.mdEventStart = result.params['mu1'].value
                    self.mdEventEnd = result.params['mu2'].value
                    self.mdBlockDepth = self.mdBlockedCurrent / self.mdOpenChCurrent
                    self.mdResTime = self.mdEventEnd - self.mdEventStart
        except:
            # print optfit.message, optfit.lmdif_message
            self.flag = -2
            self.mdBlockDepth = -2
            self.mdResTime = -2
            print('error')


class BlockageDetector:
    #  实时维护队列，保证Windows_size的数据是最新的基线数据，判断插入的数据是否是阻塞数据
    def __init__(self, sigma, window_size=50000, scale=12):
        self.window_size = window_size
        self.scale = scale
        self.signal_queue = deque(maxlen=window_size)
        self.baseline = None
        self.std_dev = sigma
        self.threshold = None
        self.blocked_segments = []
        self.threshold_segment = []
        self.baseline_segment = []
        self.block_flag = []
        self.BlockDepth = []
        self.resDwellTime = []
        self.filter = []
        self.peak_value = []
        self.mean_value = []
        self.std_value = []
        self.globtime_ = []

    def __getitem__(self, idx):
        baseline = self.baseline_segment[idx]
        threshold = self.threshold_segment[idx]
        blocked_segments = self.blocked_segments[idx]
        block_flag = self.block_flag[idx]
        block_depth = self.BlockDepth[idx]
        resDwellTime = self.resDwellTime[idx]
        filter = self.filter[idx]
        return baseline, threshold, blocked_segments, block_flag, block_depth, resDwellTime, filter

    def update_baseline(self, new_signal):
        if len(self.signal_queue) == self.window_size:
            # 发生阻塞
            if self.threshold > new_signal or (self.baseline + self.scale * self.std_dev) < new_signal:
                return True
            else:
                # 添加基线判断
                if (new_signal > (self.baseline - 2 * self.std_dev)) and (
                        new_signal < (self.baseline + 2 * self.std_dev)):
                    removed_signal = self.signal_queue.popleft()
                    self.signal_queue.append(new_signal)
                    self.baseline = (self.baseline * self.window_size - removed_signal + new_signal) / self.window_size
                    self.threshold = self.baseline - self.scale * self.std_dev
                return False
        if len(self.signal_queue) < self.window_size:
            self.signal_queue.append(new_signal)
            self.baseline = np.mean(self.signal_queue)
            self.threshold = self.baseline - self.scale * self.std_dev
            return False

    def is_blocked(self, signal):
        if self.baseline is None:
            return False
        if (signal > (self.baseline - 2 * self.std_dev)) and (signal < (self.baseline + 2 * self.std_dev)):
            return True
        else:
            return

    def fit_gaosi(self, data, bins):
        hx, xedge = np.histogram(data, bins)
        xedge = (xedge[1:] + xedge[:-1]) / 2
        g_init = models.Gaussian1D(amplitude=np.max(hx), mean=np.mean(data), stddev=np.std(data))
        fit_g = fitting.LevMarLSQFitter()
        g = fit_g(g_init, xedge, hx)
        return g.mean.value, g.stddev.value, g


    def process_signal_data(self, signal_data, limit):
        """
        在信号数据中检测连续的阻塞事件，然后将这些事件的起始和结束索引记录在 blocked_segments 列表中
        :param signal_data: 一段信号，阻塞信号
        :return:
        """
        for idx, signal in enumerate(signal_data):
            # 如果 update_baseline 方法返回 True，这意味着发生阻塞，队列已满且移动基线和标准差已更新，这样变动的阈值也随之更新。
            temp = self.update_baseline(signal)
            if temp:
                # 这一行调用 update_baseline 方法来更新移动基线和标准差。如果队列尚未达到 window_size，则该方法将仅将当前信号添加到队列中
                # 如果 blocked_segments 列表为空，或者上一个阻塞段的结束索引不是当前信号的前一个索引，那么会创建一个新的阻塞段。
                if not self.blocked_segments or self.blocked_segments[-1][1] != idx - 1:
                    # 如果需要创建新的阻塞段，那么就在 blocked_segments 列表中添加一个包含当前信号索引的新阻塞段
                    self.blocked_segments.append([idx, idx])
                    self.baseline_segment.append(self.baseline)
                    self.threshold_segment.append(self.threshold)
                    if self.blocked_segments.__len__() >= 2:
                        # 使用等效电路法
                        row = self.blocked_segments[-2]
                        uData = signal_data[row[0] + 3: row[1] - 3]
                        globtime = (row[0] + 3) / 1e2
                        self.globtime_.append(globtime)
                        fit = adept2State((row[0] - 50) / 1e2, signal_data[row[0] - 50: row[1] + 50], self.baseline,
                                          self.std_dev)
                        fit.FitEvent()
                        # temp = temp[(temp['DwellTime'] >= DwellMin) & (temp['DwellTime'] <= DwellMax) & (temp['BlockDepth'] > BlockDepthMin) & (temp['BlockDepth'] <= BlockDepthMax)]
                        if (fit.mdResTime >= limit.DwellMin) & (fit.mdResTime <= limit.DwellMax):
                            try:
                                y, x = np.histogram(uData, bins=100)
                                popt, pcov = curve_fit(gaussian, x[:-1], y)
                                self.peak_value.append(popt[0])
                                self.mean_value.append(popt[1])
                                self.std_value.append(popt[2])
                            except:
                                self.peak_value.append(0)
                                self.mean_value.append(0)
                                self.std_value.append(0)
                            self.block_flag.append(1)
                            self.BlockDepth.append(fit.mdBlockDepth)
                            self.resDwellTime.append(fit.mdResTime)
                        else:
                            self.block_flag.append(0)
                            self.peak_value.append(0)
                            self.mean_value.append(0)
                            self.std_value.append(0)
                            self.BlockDepth.append(fit.mdBlockDepth)
                            self.resDwellTime.append(fit.mdResTime)
                            # 不满足过滤条件直接去除那个阈值法的事件
#                             self.blocked_segments.pop(-2)
#                             self.threshold_segment.pop(-2)
#                             self.baseline_segment.pop(-2)
                else:
                    # 在这种情况下，将更新上一个阻塞段的结束索引为当前信号的索引
                    self.blocked_segments[-1][1] = idx
        # 将信号最后的信号清除
        self.blocked_segments.pop()
        self.threshold_segment.pop()
        self.baseline_segment.pop()


# 高斯拟合函数
def gaussian(x, a, s, m):
    return a * np.exp(-(x - m) ** 2 / (2. * s ** 2))


def get_gauss_sigma(signal):
    x = np.arange(len(signal))
    popt, pcov = curve_fit(gaussian, x, signal)
    return popt, pcov


def get_signal_mean_sigmal(dat, limit, minBaseline=-1, maxBaseline=-1):
    """
    计算一个时间序列数据的均值和标准差，并根据特定的限制条件进行高斯拟合。
    dat：时间序列数据。
    limit：限制计算范围的参数，取值为 0.5（限制在最大值和最小值的 50% 到 50% 之间）、-0.5（限制在最小值和最大值的 50% 到 50% 之间）或 0（整个范围）。其他非零值将重置为 0。
    minBaseline：最小基线值，默认值为 -1。
    maxBaseline：最大基线值，默认值为 -1。
    """
    datsign = np.sign(np.mean(dat))
    uDat = datsign * dat
    dMin, dMax, dMean, dStd = np.floor(np.min(uDat)), np.ceil(np.max(uDat)), np.round(np.mean(uDat)), np.std(uDat)
    try:
        hLimit = {0.5: [dMean, dMax], -0.5: [dMin, dMean], 0: [dMin, dMax]}[limit]
    except KeyError:
        hLimit = [dMin, dMax]

    if minBaseline == -1.0 or maxBaseline == -1.0:
        y, x = np.histogram(uDat, range=hLimit, bins=100)

    else:
        hLimit = [minBaseline, maxBaseline]
        y, x = np.histogram(uDat, range=hLimit, bins=100)
    try:
        sigma = 1 / np.sqrt(y + 1e-10)
        popt, pcov = curve_fit(gaussian, x[:-1], y, p0=[np.max(y), dStd, np.mean(x)], sigma=sigma)
        perr = np.sqrt(np.diag(pcov))
    except:
        return [0, 0]
    if np.any(perr / popt > 0.5) or ((minBaseline > -1 and maxBaseline > -1) and (popt[2] < minBaseline or popt[
        2] > maxBaseline)):  # 0.5 is arbitrary for the moment, for testing. Could be added as a parameter or hard-coded pending testing.
        return [0, 0]

    return [popt[2], np.abs(popt[1])]


def filterBolcktime(filename, minLen, maxLen):
    """
    过滤阻塞时长
    :param filename: 需要处理的文件名
    :param minLen: 最小点的个数
    :param maxLen: 最多点的个数
    :return:
    """
    data = pd.read_json(filename)
    res = data[data["blockedSegment"].apply(lambda x: not ((x[1] - x[0] < minLen) or (x[1] - x[0] > maxLen)))]
    return res


def heaviside(x):
    """

    :param x:
    :return:
    """
    out = np.array(x)
    out[out == 0] = 0.5
    out[out < 0] = 0
    out[out > 0] = 1

    return out


def stepResponseFunc(t, tau1, tau2, mu1, mu2, a, b):
    """
    等效电路法
    :param t:
    :param tau1:
    :param tau2:
    :param mu1:
    :param mu2:
    :param a:
    :param b:
    :return:
    """
    try:
        t1 = (np.exp((mu1 - t) / tau1) - 1) * heaviside(t - mu1)
        t2 = (1 - np.exp((mu2 - t) / tau2)) * heaviside(t - mu2)
        # Either t1, t2 or both could contain NaN due to fixed precision arithmetic errors.
        # In this case, we can set those values to zero.
        t1[np.isnan(t1)] = 0
        t2[np.isnan(t2)] = 0

        return a * (t1 + t2) + b
    except:
        raise


def processSignalSegment(filename, scale, limit):
    """
    输入对应的文件路径，获取该文件对应的阻塞事件字典集合
    :param filename: 文件名
    :param scale 阈值法的缩放尺度
    :param limit 这个是过滤条件
    :return:
    """
    abf_file_path = filename + ".abf"

    # 使用pyabf打开abf文件
    abf = pyabf.ABF(abf_file_path)

    # 获取信道数量
    num_channels = abf.channelCount

    # 获取采样率
    sampling_rate = abf.dataRate

    # 获取数据
    data = abf.data
    # 使用默认的0.5 s
    signal_data_ = np.array(data[0][0:50000])
    # signal_data = np.array(data[0][1200::])
    signal_data = np.array(data[0])

    # 根据前三秒获取方差
    mu, sig = get_signal_mean_sigmal(signal_data_, 0.5)

    # 根据前面获得的高斯方差进行构建阻塞事件识别对象
    detector = BlockageDetector(sigma=sig, window_size=1000, scale=scale)
    # 处理数据
    detector.process_signal_data(signal_data, limit)

    signal_dict = {
        'baseline': detector.baseline_segment, 'blockedSegment': detector.blocked_segments,
        'threshold': detector.threshold_segment, 'blockDepth': detector.BlockDepth,
        'blockFlag': detector.block_flag, 'dwellTime': detector.resDwellTime
    }
    res = pd.DataFrame(signal_dict)
    res.to_json(filename + "filterNot.json", orient="records")
    signal_dict["global time"] = detector.globtime_
    signal_dict["peak"] = detector.peak_value
    signal_dict["std"] = detector.std_value
    res = pd.DataFrame(signal_dict)
    res.to_excel(filename + ".xlsx")
    return res

def main1(filename, scale, limit):
    """
    根据输入事件路径，完成阻塞识别，时间过滤，阻塞率计算
    :param filename:
    :return:
    """
    now_time = time.time()
    print('Start ......')
    # 事件识别
    processSignalSegment(filename, scale, limit)
    end_time = time.time()
    print(f"{filename} 阻塞事件识别耗时：{(end_time - now_time) / 60}min")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--DwellMin', type=int, default=0)
    parser.add_argument('--DwellMax', type=int, default=1000)
    parser.add_argument('--savepath', type=str, default=None)
    parser.add_argument('--inputfile', type=str, default='../data/train/Rb1')
    args = parser.parse_args()
    path = args.inputfile
    for filename in os.listdir(path):
        file_paths = os.path.join(path, filename)
        files = []
        for filename in glob(file_paths + '/*.abf'):
            files.append(filename.strip('.abf'))
        process_list = []
        # 将thread_num个任务分配给进程池中的进程执行
        for i in range(len(files)):
            p = Process(target=main1, args=('..' + files[i], 12, args,))
            p.start()
            process_list.append(p)
        for p in process_list:
            p.join()
