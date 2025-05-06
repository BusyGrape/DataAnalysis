# [时间序列Time Series](https://www.kaggle.com/learn/time-series "")

如何分析和预测以时间或发生次序作为索引的数据。这样的数据可以分析它的趋势Trend，周期性Seasonality，时序依赖性Serial Dependence。

与非预测类型的算法结合使用Hybrid Models，让模型的预测兼具趋势展望和各类影响因素的反馈两重能力。

以时间作为索引的数据，可以用deterministic将时间处理为多组数字（例如用n代表n个单位时间来计算趋势），方便作为自变量X来使用。
```Python
from statsmodels.tsa.deterministic import DeterministicProcess

dp = DeterministicProcess(
     index=df.index,      # dates from the training data
     constant=True,       # dummy feature for the bias (y_intercept)
     order=1,             # 1 for linear, 2 for quadratic, 3 for cubic, and so on.
     drop=True,           # drop terms if necessary to avoid collinearity
)
```
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
用线性回归拟合出趋势线，并预测数据未来走势
```Python
from sklearn.linear_model import LinearRegression

y = df['target_value']

# `in_sample` can creates features for the dates given in the `index` argument
X = dp.in_sample()    

# The intercept is the same as the `const` feature from DeterministicProcess.
# LinearRegression behaves badly with duplicated features,
# so we need to be sure to exclude it here.
model = LinearRegression(fit_intercept=False)
model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=X.index)

# `out_of_sample` can creates features outside of the period of the training data
X = dp.out_of_sample(steps=30)  # make a 30-day forecast

y_fore = pd.Series(model.predict(X), index=X.index) 
```

在同一个图上画出训练数据data（点），趋势trend（线）和预测走势trend forecast（线）
```python
import matplotlib.pyplot as plt

# **plot_parms indicate a series setting for plot
# plot_params = dict(color=, style=, etc.)
# which are skipped here
ax = df["start date":].plot(title=" - Linear Trend Forecast", **plot_params)
ax = y_pred["start date":].plot(ax=ax, linewidth=3, label="Trend")
ax = y_fore.plot(ax=ax, linewidth=3, label="Trend Forecast", color="C3")
_ = ax.legend()
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
