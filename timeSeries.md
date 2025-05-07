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

y = df['target_val']

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

We will learn two kinds of features that model seasonality. 

The first kind, indicators, is best for a season with few observations, like a weekly season of daily observations. 
指针归类：周期比较短，重复出现次数多。比如一年里的52个周。周一到周日的数据呈现相同的趋势。画在一个图里时，不同周的数据可能高低不一样，但是以同样的规律变化。

The second kind, Fourier features, is best for a season with many observations, like an annual season of daily observations.
傅里叶周期：在一个较长的时间里可以观测到有不定数量的波浪出现。所有周期性函数都可以用若干组傅里叶曲线来表达。

因此问题被拆解成两个部分：
1. 在较长的时间范围里，找到合适数量的傅里叶曲线来描述大周期变化

   关于回答，应该用多少组傅里叶曲线来描述大周期。可以通过观察periodogram，找到最后一个波峰
```Python
plot_periodogram(df.target_value)

from statsmodels.tsa.deterministic import CalendarFourier

# Computing Fourier features used as `additional_terms` in `DeterministicProcess()`
fourier = CalendarFourier(freq="A",    # A=annual, M=monthly
                          order=10)    # 10 sin/cos pairs for "A"nnual seasonality
```
2. 再描述小的周期里数据是根据不同指针的位置是如何变化的

   在回归分析过程里，通过带入一组经过独热编码的周天指针来实现

```Python
from statsmodels.tsa.deterministic import DeterministicProcess

dp = DeterministicProcess(
     index=tunnel.index,
     constant=True,               # dummy feature for bias (y-intercept)
     order=1,                     # trend (order 1 means linear)
     drop=True,                   # drop terms to avoid collinearity

     # only for seasonality
     seasonal=True,               # weekly seasonality (indicators)
     additional_terms=[fourier],  # annual seasonality (fourier)
)

```
回归分析建模以及绘图都和trend相同，略

## 时序依赖性Serial Dependence

数据的时序规律体现在和上1（n）步或者下1（n）步的关系是有规律的。这种规律展现的是下一期数据的走势和前期数据高度相关cyclic，但是如果拿来跟同期比较，就看不出什么规律。

观察似乎否有时序依赖性，要把数据往前或往后错1步或者n步，然后用移位后的数据Lagged Features作为自变量X
```Python
import pandas as pd

# use column.shift(n) n=steps count to create lag feature
df[val_lag_1]=df[val].shift(1)

# define a function to generate several lagged features
def make_lags(ts, lags):
    return pd.concat(
        {
            f'y_lag_{i}': ts.shift(i)
            for i in range(1, lags + 1)
        },
        axis=1)

# get lagged features
X = make_lags(df.target_val, lags=4)
X = X.fillna(0.0)
```
如果跟多个步之前的数据比较，中间几步要不要也带入模型，它们之间有没有互相影？互相影响的X非常不适合都带入线性回归模型。可以通过partial autocorrelation，和散点图来判断（对参考其他课程： mutual information）
```Python
from statsmodels.graphics.tsaplots import plot_pacf

_ = plot_pacf(df.target_val, lags=12)

# 需要手动写一个画图公式，略
_ = plot_lags(df.target_val, lags=12, nrows=2)

```

## 混合模型Hybrid Models
There are generally two ways a regression algorithm can make predictions: either by transforming the features or by transforming the target. 

Feature-transforming algorithms learn some mathematical function that takes features as an input and then combines and transforms them to produce an output that matches the target values in the training set. Linear regression and neural nets are of this kind.

Target-transforming algorithms use the features to group the target values in the training set and make predictions by averaging values in a group; a set of feature just indicates which group to average. Decision trees and nearest neighbors are of this kind.

### Components and Residuals

基本结合思路，组合公式：<br>
series = trend + seasons + cycles + error

找出以上所有和时序有关的规律，构建时序特征变量，建模拟合；然后还有预测不了的差值，丢给其他模型处理。

建模步骤
```
# 1. Train and predict with first model
model_1.fit(X_train_1, y_train)
y_pred_1 = model_1.predict(X_train)

# 2. Train and predict with second model on residuals
model_2.fit(X_train_2, y_train - y_pred_1)
y_pred_2 = model_2.predict(X_train_2)

# 3. Add to get overall predictions
y_pred = y_pred_1 + y_pred_2
```

## Forecasting With Machine Learning
