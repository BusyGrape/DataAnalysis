# Regression Analysis 回归分析
简单介绍做“回归分析”的步骤PACE：从拆解问题到结论释义<br>
学习两种回归分析方法linear and logistic：线性回归和逻辑回归<br>
了解这两种回归分析如何解决实际问题。

## M1 Introduction
### 回归分析四步法 PACE
P 计划 plan——充分了解项目需求和可用资源<br>
A 分析 Analyze——试试哪个模型更适合用来做回归分析<br>
C 建模 Construct——建模并处理数据<br>
E 执行 Execute——根据运算结果进行分析，得到结论，并解释

### 线性回归 liner regression
适合用来描述：<br>
一种因素Y会跟随着另一种因素X的变化而改变。<br>
他们之间的伴随关系是线性的，也就是说，Y会按某种比例，随X的增加而增加（或减少）<br>
μ{Y|X} = β<sub>0</sub>+β<sub>1</sub>X
- Y，Dependent variable 
- X，Independent variable
- Regression coefficients
	- β<sub>1</sub>，Slope
	- β<sub>0</sub>，Intercept
- OLS，ordinary least squares estimation
	- loss functiion用公式推算出的理想Y与实际观测到的Y之间的差

### 逻辑回归 logistic regression
