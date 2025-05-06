# [时间序列Time Series](https://www.kaggle.com/learn/time-series "")

如何分析和预测有用时间或发生次序索引的数据。可以分析趋势Trend，周期性Seasonality，时序依赖性Serial Dependence。

与非预测类型的算法结合使用Hybrid Models，让模型的预测兼具趋势展望和各类影响因素的反馈两重能力。

## 趋势Trend
在短时间内数据有很大波动，但是长期看呈现某种走势。

可以用一个更平滑的曲线表达和观察这种长期走势。比如以一个固定周期计算的移动平均值（汇总值，最小值，最大值，平方差等）。

```Python
import pandas as pd

moving_average = df.rolling(
    window=365,       # size of the rolling window
    center=True,      # whether to put the average at the center of the window
    min_periods=183,  # minimum number within a rolling window that are required to have a non-NaN result
).mean()              # compute the mean (could also do median, std, min, max, ...)
```
在同一个图上画出实际数据（点）和趋势（线）
```python
ax = df.plot(style=".", color="0.5")    # plot the series of data
moving_average.plot(    
    ax=ax,                              # plot the trend at the same ax/position
    linewidth=3,
    title="365-Day Moving Average",
    legend=False,
)
```

## 周期性Seasonality
周期性函数可以用若干组傅里叶曲线来表达。

```Python

```

## 时序依赖性Serial Dependence

## Hybrid Models
There are generally two ways a regression algorithm can make predictions: either by transforming the features or by transforming the target. 

Feature-transforming algorithms learn some mathematical function that takes features as an input and then combines and transforms them to produce an output that matches the target values in the training set. Linear regression and neural nets are of this kind.

Target-transforming algorithms use the features to group the target values in the training set and make predictions by averaging values in a group; a set of feature just indicates which group to average. Decision trees and nearest neighbors are of this kind.
