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
他们之间的伴随关系是线性的，也就是说，Y会按某种比例，随X的增加而增加（或减少）

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
Y的归类是可以计数的，从两类到有限多类。

μ{Y|X} = Prob(Y = n|X) = p

- p，probability 当X是某个值的时候，Y属于第n种归类的可能性是多少。
- link function 推算逻辑回归公示的方法

### Correlation is not Causation 

两种因素相关，并不等于谁能引发谁。这不是一种因果关系。

## M2 Simple linear regression 一元线性回归模型

Y的变化只受一个因素X影响。<br>
在一个二维坐标系里，每对相关因素X和Y的关系能折射成一个点。找出一条直线（的方程），最能表示这些点的分布趋势。

### OLS 最小误差法推算回归方程

minimizing loss function / error

- ε，Residual 残差，Y观测值 与 Y的推测值 之间的差异

	Σ(ε) == 0 使用OLS法，残差和永远为零

- SSR，sum of squared residuals 残差平方和
- 在所有的直线选项里，找那条让SSR值最低的（线）
	
### 用r计算回归线方程
- r，Pearson's correlation / linear correlation coefficient 线性相关系数	

	- r在[-1,1]之间

		r的绝对值越大，样本点越像一条直线；r越小，样本点越接近一团无序云状散点

	- r 正负的含义

		r>0意味着斜率是正数，倾斜角度↗<br>
		r<0意味着斜率是负数，倾斜角度↘

	- r的数值不是斜率
	- r = covariance(X,Y)协方差/(SD X)(SD Y)标准差
	
- 有两条定律
	- X的平均值和Y的平均值永远会落在回归线上
	- 如果X增加1个X标准差，那么Y会增加r个Y标准差
	
- 回归线的斜率β<sub>1</sub> = r(SD Y)/SD X
		
### 用Python来完成所有计算
- A 预分析阶段
	- 观察两两散点图矩阵
			
		```python
		# 我搜了一下，多个帖子都说
		# 画两两散点图就用seaborn库
		import seaborn as sns		
		# 给每两个变量之间画一幅散点图
		sns.pairplot(origData)
		```

- C 建模
	- Step1 Build a model
			
		```python
		# Subset Data 清洗并选择要进行回归分析的两列数据
		ols_data = origData[["column1/X", "column2/Y"]]
		# Write out formula 定义Y和X分别是哪列数据
		ols_formula = "column2/Y ~ column1/X"
		# Import ols function
		from statsmodels.formula.api import ols
		# Build OLS, fit model to data 用OLS方法建模计算出回归线
		OLS = ols(formula = ols_formula, data = ols_data)
		model = OLS.fit()
		```
							
	- Step 2 Model evaluation

		P-value，Confidence Intervals

		```python
		# print statistics 输出模型的各项统计指标
		model.summary()

		# confidence band 
		sns.regplot(x="column1/X", y="column2/Y", data = ols_data)
		
		# X，取X值
		X = ols_data["column1/X"]
		# Y，用预测公式Predict获得Y值fitted_values
		fitted_values = model.predict(X)
		# Residuals，用resid公式获得残差值
		residuals = model.resid
		```
		
		Homoscedasticity

		```python
		# Residuals在0附近的偏移量散点图
		import matplotlib.pyplot as plt
		fig = sns.scatterplot(x=fitted_values, y=residuals)
		fig.axhline(0)
		fig.set_xlabel("Fitted Values")
		fig.set_ylabel("Residuals")
		plt.show()
		```
		
		Normality

		```python
		# Residuals的柱状图
		fig = sns.histplot(residuals)
		fig.set_xlabel("Residual Value")
		fig.set_title("Histogram of Residuals")
		plt.show()
		# Q-Q plot图	
		import statsmodels.api as sm
		fig = sm.qqplot(model.resid, line = 's')
		plt.show()
		```
			
		R<up>2</up>，MSE/MAE
		```python
		# 导入库sklearn
		from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
		# 假设已经有一组Hold_out sample[["x_test","y_test"]]
		y_test_pred = model.predict(x_test)
		MSE = mean_squared_error(y_test,y_test_pred)
		MAE = mean_absolute_error(y_test,y_test_pred)
		R2 = r2_score(y_test,y_test_pred)
		print('MSE:',MSE,'/nMAE:',MAE,'/nR2:',R2)
		```		
		
- E 可以用python来完成Data visualisation工作
		
### 建模分析过程PACE

- A 检查线性回归假设是不是都满足
	- Linearity 两个变量XY之间是否是线性相关

		可以先抓样本画个散点图，看看这些点的分布像不像直线
	
	- Normality 残差residual是否是normal distributed正态分布

		必须要建完模型后才能检验，可以直接画柱状图也可以画Q-Q plot图来观察判断

	- Independent observations 采集的样本之间没有相互影响
		主要靠分析数据采集步骤来判断，或者建模后输出结果显示有问题时能发现样本不符合假设
	
	- Homoscedasticity 残差的偏移量持续且存在且数值随机但接近

		必须要建完模型后才能检验，同样可以通过画出散点图来观察判断
		
- C 用适合的变量建模，得到各项统计数据
	- 建模（参见：用Python来完成所有计算/C 建模）
	- 评估
		
		了解summary给出的各个指标都是什么意思
		- confidence band
		
			在回归线附近，落在置信区间里所有的直线组成的一片区域<br>
			用样本进行回归分析，总是可能与整体的回归分析结果存在偏差
		
		- P-value			
		
			null hypothesis（XY不存在线性相关性）成立的可能性（概率）<br>
			如果P小于置信度（5%），null hypothesis被推翻<br>
			可以认为XY之间线性相关，coefficient不是0
			
		- Confidence Intervals [0.025, 0.975]
		
			表示有5%的机会，斜率和截距的置信范围值不能包含回归线的真实参数<br>
			在这一列下面给出来的数据是截距和斜率的范围，用来画出cofidence band区域
		
		常见的评估矩阵
		- R<sup>2</sup> 决定系数 0~1之间
			
			用来描述X对Y的影响程度。越接近于1说明越适合用线性回归分析。
			线性相关系数r的平方(有待验证，计算公式不一样)
			
		- MSE，mean squared error
			
			对outlier敏感，值越小越好
			
		- MAE，mean absolute error 
			
			在有outlier时使用，值越小越好 
			
		- Hold-out sample 
			
			不能是之前建模fit model时使用过的数据，可以用这组新数据检测以上三个值
			
- E 对建模结果进行全面评估和解释
	- 解释模型的统计指标都意味着什么，比如斜率表明的Y将如何因为X而变化
	- 尽量将数字转化成易于理解的图像、动画等来进行讲解或演示
	- 有必要提醒听众模型仍可能在什么情况下失效，或需要进行修正
	- 但要尽量避免使用晦涩的术语，如coefficients or P-value 
	- 要注意区分correlation和causation的区别，我们几乎无法在这里证明causaiton

## M3 Multiple linear regression 多元线性回归模型

当不止一个因素共同影响Y时，引入多元线性回归模型

Y = β<sub>0</sub>+β<sub>1</sub>X<sub>1</sub>+...+β<sub>n</sub>X<sub>n</sub>

### One hot encoding 独热编码

如果影响因素X<sub>i</sub>是分类变量。类似多选题的答案，如：做或者不做，选择了AC没有选BDE。<br>
这时为了可以进行回归分析，要把X<sub>i</sub>拆解成一组二进制数来表示它的全部特征<br>
X<sub>i</sub>→X<sub>iA</sub>,X<sub>iB</sub>,...,X<sub>iN</sub><br>
二进制的位数和X有几个特征有关

### Interaction 交互项/交叉变量

这个概念我觉得本课说的不很清楚，看完后我还有好几个疑问<br>
什么时候加入交互项？不论影响因素是分类变量还是连续数值变量都可以引入交互项么？如何解读交互项？
最后找了一篇知乎文章算是基本看明白了，配合评论区就更全面的回答了我的疑问。

[一文轻松看懂交互作用](https://zhuanlan.zhihu.com/p/224990519 "")

### 多元线性回归假设
- Linearity 
	
	看每一个X<sub>i</sub>和Y的散点图是不是像一条线
	
- Independance, Normality, Homoscedasticity
	
	定义和检测都和一元线性回归假设一样
	
- No multicollinerity assumption 自变量之间没有线性关系假设

	通过所有变量之间的两两散点图来判断

	```python
	sns.pairplot()
	```
	如果散点图不好判断，可以计算两个变量之间的VIF值（1~∞）。VIF越大线性关系越强。
	
	```python
	from statsmodels.stats.outliers_influence import variance_inflation_factor
	X = df[['col_1', 'col_2', 'col_3']]
	vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
	vif = zip(X, vif)
	print(list(vif))
	```
	避免同时挑选上两个明显有线性关系的变量作为X<sub>i</sub>&X<sub>j</sub>，或是将两个有很强线性关系的变量转化成一个新的变量。
		
### 用python建立多元回归模型
- C 建模

	```python
	# 准备数据
	X = origData[["col_1/X1","col_2/X2",...,"col_n/Xn"]]
	Y = origData[["col_0/Y"]]
	# 导入库
	from sklearn.model_selection import train_test_split
	# 把数据分成建模和测试两部分
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42]
	# 准备建模用api
	ols_data = pd.concat([X_train, y_train], axis = 1)
	# Write out formula 定义Y和X分别是哪列数据
	ols_formula = "col_0/Y ~ col_1/X1 + C(col_2/categorical X2)+...+col_n/Xn"
	# Import ols function
	from statsmodels.formula.api import ols
	# Build OLS, fit model to data 用OLS方法建模计算出回归线
	OLS = ols(formula = ols_formula, data = ols_data)
	model = OLS.fit()	
	```

- E 各项统计指标的含义

### variable selection

选择包含什么特征/影响因素/自变量到回归模型里。根据建模后的指标数值，调整模型。

- underfitting和overfitting
	
	R<sup>2</sup>太低或太低或太高<br>
	太低等于回归模型没有抓住样本的特征，也就约等于拟合无效<br>
	太高则有可能是因为太贴合样本的特征，所以反而无法延展出整体的特征，不能很好地用于预测未知数据组
	
- Adjusted R-squared value

	R<sup>2</sup>会随着样本数量增加而自然趋近于1，Ajusted R<sup>2</sup>去除了样本数量和特征（自变量）数量对评分的影响，所以更好用。

- 常见筛选方法
	- forward selection & backward elimiation
	
		forward是从第一个可能的特征/因素/自变量开始，一个一个判断是否要包含<br>
		backward是先包含所有可能的特征，再从最后一个开始判断是不是要剔除

	- based on Extra-sum-of-squares F-test
		
		based on p-value

- Regularization 正则化
	
	解决过度拟合的模型，降低variance增加bias<br>
	Lasso Regression 去掉所有对预测Y不太有用的特征<br>
	Ridge Regression 降低不重要特征的影响但不会去掉它们
	
	Elastic Net regression 测试lasso和ridge哪个或混合模式更好
	
	Principal component analysis (PCA) 阅读材料里的概念

## M4 Advanced hypothesis testing 假设检验
### chi-squared test χ<sup>2</sup>卡方检验

用于检验与分类变量相关的假设

- χ<sup>2</sup> Goodness of fit test 卡方适合度检验

	观测值是否符合预期分布规律<br>
	Null hypothesis(H<sub>0</sub>) 观测值符合预期分布规律<br>
	Alternative hypothesis(H<sub>1</sub>) 不符合预期分布
	
	卡方值计算公式<br>
	χ<sup>2</sup> = Σ((observed-expected)<sup>2</sup>/expedted)
	
	再继续若干步计算后查出P-value，根据置信度决定拒绝或接受H<sub>0</sub><br>
	P<置信度，拒绝H<sub>0</sub>
	
	expected values分类数小于5时，适合度检验可能不准确。
	
	```python
	import scipy.stats as stats
	observations = [650, 570, 420, 480, 510, 380, 490]
	expectations = [500, 500, 500, 500, 500, 500, 500]
	result = stats.chisquare(f_obs=observations, f_exp=expectations)
	```
	
- χ<sup>2</sup> Test for independence 卡方独立性检验

	检验两个分类变量之间是否相互独立<br>
	Null hypothesis(H<sub>0</sub>) 两个分类变量之间相互独立<br>
	Alternative hypothesis(H<sub>1</sub>) 两个分类变量互相影响
	
	矩阵，横列是变量1的若干种情况，纵列是变量2的若干种情况<br>
	横纵交叉点上记录观测数量的累计值<br>
	再计算每个交叉点的期望值E<sub>ij</sub> = R<sub>i</sub>C<sub>j</sub>/T
	
