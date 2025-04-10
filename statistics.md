# The Power of Statistics

[Course link](https://www.coursera.org/learn/the-power-of-statistics/home/module/1"") 

学习进行数据分析所需要的统计学知识。

## M1 Introduction to Statistics with Python

### Descriptive Statistics 描述性统计

Describe or summarize the main features of a dataset
描述或总结一组数据的特征

- Measures of central tendancy 集中趋势度量
	- mean 中位数
	- median 平均数
	- mode 众数

- Measures of dipersion 离散趋势度量/离散量
	- range 极差
	- variance 方差
	- standard deviation 标准差

- Measures of position 位置度量
	- percentiles 百分位数P<sub>k</sub>
	- quartiles 四分位数 Q1=25/Q2=50%/Q3=75%
	- interquartile range 四分位距IQR=Q3-Q1
	
### Inferential Statistics 推断性统计

Make inferences about a dataset based on a sample of the data
由样本推断整体

- parameter and statistic 参数 统计量
	
	Parameter is a characteristic of a <b>population</b>
	
	Statistic is a characteristic of a <b>sample</b>
	
- population and sample 总体 样本

	Population includes every possible element that you are interested in measuring
	
	Sample is subset of population
	
	Representative sample is a sample accurately reflects the population
	
- difference between descriptive statistics and inferential statistics

	Descriptive statistics let you quickly understand a large amout of data. If you summarize the data, you can instantly make it meaningful.
	
	Inferential statistics allow you draw conclusions and make predictions based on a sample of the data.

## M2 Probablity

Probability uses math to quantify uncertainty, or to describe the likelihood of something happening.

### Foundational concepts
- Concepts
	- Random experiment 随机试验, a process whose out come cannot be predicted with certainty
	- Outcome 试验结果, the result of a random experiment
	- Event 事件, a set of one or more outcomes

- Three basic rules
	- Complement rule 互补定律
	
		mutually exclusive events: A and A<sup>'</sup><br>
		P(A<sup>'</sup>)=1-P(A)
		
	- Addition rule(for mutually exclusive events)
		
		mutually exclusive events: A and B
		P(A or B) = P(A)+P(B)
		
	- Multiplication rule(for independent events)
		
		independent events: A and B
		P(A and B) = P(A)×P(B)

### Conditional probability 条件概率
- Dependent events, the occurrence of one event changes the probability of the other event
- Conditional probability
	
	P(A and B) = P(A) * P(B|A)<br>
	P(B|A) = P(A and B) / P(A)<br>
	the occurrence of event B depends on the occurrence of event A.

- Bayes’s theorem 贝叶斯定理
	- Posterior and prior probability
		
		Prior probability refers to the probability of an event before new data is collected. Posterior probability is the updated probability of an event based on new data.
		
		P(A): Prior probability<br>
		P(A|B): Posterior probability 

	- Fomular

		P(A|B) = P(B|A)*P(A)/P(B)
	
	- Liklihood and evidence
	
		P(B|A): Likelihood<br>
		P(B): Evidence 

### Probability distributions
- Discrete distributions 离散分布

	Discrete probability distributions represent discrete random variables, or discrete events. Often, the outcomes of discrete events are expressed as whole numbers that can be counted.
	
	- Uniform distribution 均匀分布 describes events whose outcomes are all equally likely, or have equal probability
	- Binomial distribution 二项分布 refers to repeated trials with only two possible outcomes
	- Bernoulli distribution 伯努利分布 refers to only a single trial of an experiment with only two possible outcomes
	- Poisson 泊松分布 models the probability that a certain number of events will occur during a specific time period
	
- Continuous distributions
	
	continuous probability distributions represent continuous random variables, which can take on all the possible values in a range of numbers.
	
	- Normal distribution/Gaussian distribution 正态分布/高斯分布
		
		The normal distribution is a continuous probability distribution that is symmetric about the mean and bell-shaped. It is also known as the Gaussian distribution

	- All normal distributions have the following features: 

		The shape is a bell curve<br>
		The mean is located at the center of the curve<br>
		The curve is symmetrical on both sides of the mean<br>
		The total area under the curve equals 1
	
	- Empirical rule 经验法则/3σ/68-05-99.7法则
		
		1σ-68% of values fall within 1 standard deviation of the mean<br>
		2σ-95% of values fall within 2 standard deviations of the mean<br>
		3σ-99.7% of values fall within 3 standard deviations of the mean
		
- Probability Density and Probability 
	
	A probability function is a mathematical function that provides probabilities for the possible outcomes of a random variable. 

	There are two types of probability functions: 

	- Probability Mass Functions (PMFs) represent discrete random variables
	- Probability Density Functions (PDFs) represent continuous random variables 
	
### Probability distributions with python

## M3 Sampling

### Sampling process

1. Identify the target population 根据要分析的事情选择有用的特征界定数据范围
2. Select the sampling frame 选择一种抽样的依据，比如用名字/用序号抽样
3. Choose the sampling method 
4. Determine the sample size 
5. Collect the sample data

### Sampling methods

- Probability sampling, uses random selection to generate a sample
	- Simple random sampling 直接从整体里随机抽样
	- Stratified random sampling 分层抽样 先给整体进行分组，再从每个分组里随机抽样
	- Cluster random sampling 整群抽样 给整体分组，然后随机选一组作为样本
	- Systematic random sampling 等距抽样 给整体排序，然后随机选一个起始点和间隔长度取样
	
- Non-probability sampling is often based on convenience
	- Convenience sampling
	- Voluntary response sampling
	- Snowball sampling 滚雪球抽样 随机选则被访问者，再由他们提供其他调研对象
	- Purposive sampling
	
### Sampling distributions
- Central limit theorem 

	The sampling distribution of the mean approaches a normal distribution as the sample size increases. And, as you sample more observations from a population, the sample mean gets closer to the population mean. 

- Conditions
	- Randomization
		
		Your sample data must be the result of random selection. Random selection means that every member in the population has an equal chance of being chosen for the sample.

	- Independence
		
		Your sample values must be independent of each other. Independence means that the value of one observation does not affect the value of another observation. Typically, if you know that the individuals or items in your dataset were selected randomly, you can also assume independence.

		- 10%: To help ensure that the condition of independence is met, your sample size should be no larger than 10% of the total population when the sample is drawn without replacement (which is usually the case). 

	- Sample size: The sample size needs to be sufficiently large.

	In general, many statisticians and data professionals consider a sample size of 30 to be sufficient when the population distribution is roughly bell-shaped, or approximately normal. However,  if the original population is not normal—for example, if it’s extremely skewed or has lots of outliers—data professionals often prefer the sample size to be a bit larger. Exploratory data analysis can help you determine how large of a sample is necessary for a given dataset. 
	
- Sampling distribution of the sample mean

- Standard error, the standard deviation of a sample statistic
	- standard error of the sample mean for a single sample
		
		standard error = s / √n
		
		s refers to the sample standard deviation(σ)<br>
		n refers to the sample size.

### Sampling distributions with Python
	 
## M4 Confidence intervals

### Describe uncertainty of estimate
- Confidence interval(Frequentist)置信区间（频率学派）

	A range of values that describes the uncertainty surrounding an estimate

	Example: 95 CI [28, 32]	
	- Sample statistic, mean/median/mode
	- Margin of error = z-score*standard error
	- Confidence level 90%/<b>95%</b>/99%
	- 9x CI [Sample statistic +- Margin of error]
	
- Credible interval(Bayesian)贝叶斯置信区间

- Interpret confidence intervals

	Technically, 95% confidence means that if you take repeated random samples from a population, and construct a confidence interval for each sample using the same method, you can expect that 95% of these intervals will capture the population mean. You can also expect that 5% of the total will not capture the population mean. 

	- Misinterpretation 1: 95% refers to the probability that the population mean falls within the constructed interval
	- Misinterpretation 2: 95% refers to the percentage of data values that fall within the interval
	- Misinterpretation 3: 95% refers to the percentage of sample means that fall within the interval
	
### Constructing confidence intervals
1. Identify a sample statistic. 
2. Choose a confidence level. 
3. Find the margin of error. 
	z-score for large sample greater than 30
	t-score for sample close to 30
	n-score for sample smaller than 30
	
	standard error SE = SD / √n
	standard error of proportion = √p(1-p)/n
	
	margin of error = x-score * SE
	
4. Calculate the interval.
	Uper limit = sample statistic + margin of error
	Lower limit = sample statistic - margin of error

### Construct a confidence intervals with Python

## M5 Introduction to hypothesis testing