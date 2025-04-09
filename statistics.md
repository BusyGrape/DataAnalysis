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

### Probability distributions with python

## M3 Sampling

## M4 Confidence intervals

## M5 Introduction to hypothesis testing