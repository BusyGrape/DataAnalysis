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

### Liner Regression 线性回归
适合用来描述：<br>
一种因素Y会跟随着另一种因素X的变化而改变。（影响因素X可以不止一种）<br>
他们之间的伴随关系是线性的，也就是说，Y会按某种比例，随X的增加而增加（或减少）<br>
μ{Y|X} = β<sub>0</sub>+β<sub>1</sub>X
- Y，Dependent variable 被影响因素
- X，Independent variable(s) 影响因素，自变量
- Regression coefficients 回归方程系数
	- β<sub>1</sub>，Slope 斜率
	- β<sub>0</sub>，Intercept 截距
- OLS，ordinary least squares estimation，推算线性回归公式的方法
	- loss functiion 理想Y与实际观测到的Y之间的差

### Logistic Regression 逻辑回归
适合用来描述：<br>
随着某种因素X的变化，Y的归类会改变。（影响因素X可以不止一种）<br>
Y的归类是可以计数的，从两类到有限多类。<br>
μ{Y|X} = Prob(Y = n|X) = p <br>
- p，probability 当X是某个值的时候，Y属于第n种归类的可能性是多少。
- link function 推算逻辑回归公示的方法

### Correlation is not Causation 
两种因素相关，并不等于谁能引发谁。这不是一种因果关系。

## M2 Simple linear regression 一元线性回归模型
在一个二维坐标系里，每对相关因素X和Y的关系能折射成一个点。找出一条直线（的方程），最能表示这些点的分布趋势。
- OLS 最小误差法
minimizing loss function / error
	- ε，Residual 残差，Y观测值 与 Y的推测值 之间的差异
	- Σ(ε) == 0 使用OLS法，残差和永远为零
	- SSR，sum of squared residuals 残差平方和
	- 在所有的直线选项里，找那条（线）让SSR值最低的
- 用r计算回归线的方程式
	- r，Pearson's correlation / linear correlation coefficient 线性相关系数	
		- r在[-1,1]之间
		r的绝对值越大，样本点越像一条直线；r越小，样本点越接近一团无序云状散点
		- r>0意味着斜率是正数，倾斜角度↗
		- r<0意味着斜率是负数，倾斜角度↘
		- r值并不能告诉我们实际斜率是什么样的
		- 计算r要用到covariance(X,Y)协方差和XY各自的SD标准差
		covariance(X,Y)/(SD X)(SD Y)
	- 有两条定律
		- X的平均值和Y的平均值永远会落在回归线上
		- 如果X增加1个X标准差，那么Y会增加r个Y标准差
		回归线的斜率就是r(SD Y)/SD X
- 用Python来完成所有计算
- 建模过程PACE

## M3 Multiple linear regression 多元线性回归模型