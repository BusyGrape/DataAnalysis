# Machine Learning 机器学习

[Course link](https://www.coursera.org/learn/the-nuts-and-bolts-of-machine-learning/home/module/1 "") 

## M1 The different types of machine learning

ML:设定好判断标准，然后让机器自己找到规律。而不是通过精确编程来实现所有的计算。

- Types of ML 机器学习种类
	- Supervised Learning

		用加过标签分类或知道含义的数据来训练模型，并用来预测
		
	- Unsupervised Learning

		分析一组没有被分类的数据中的规律

	- Reinforcement Learning
		
		通过奖励和处罚机制，让模型自行寻找最佳的（奖励最多处罚最少）策略组合
		
	- Deep Learning
		
		由不同层和节点构成，执行顺序不固定。一层执行完输送给节点判断接下来的执行的步骤，或是直接输出结论。

	- Artificial Intelligence
		
		综合使用以上所有机器学习方法

	质量比数量重要，一组数量有限但特征清晰且没有偏颇的数据，要强过于数量达标却不具备代表性的样本。
	
- Categorical VS Continuous 分类vs连续
	
	data and models need to match each other
	- Continuous
	
		Linear regression / Decision Tree regression / Lasso regression / Ridge regression / Polynomial regression / Random Forest regression / Boosting regression
	
	- Categorical/Discrete
	
		Logistic regression / Navie Bayes classfication / Decision Tree classfication / Random Forest classfication / Boosting classfication
		
- ML in everyday life 日常生活中的机器学习

	Recommendation systems

	Unsupervised learning <br>
	通过对未标记过模式的数据进行分析后给出最佳建议。量化分析事物之间的相似性，然后预测出相近选项。

	- content-based filtering 

		具体应用1：内容筛选器。<br>
		好处是：易于理解，满足用户喜好，不需要其他用户数据，可以在同一个地方对用户和物品进行画像。<br>
		坏处是：重复推荐，需要人工录入（标记）特征信息，不能跨领域迁移模型的应用。

	- collborative filtering
		
		具体应用2：共同喜好<br>
		好处是：可以跨领域迁移模型，可以发现未知的关联喜好，并不需要对待预测事物进行特征分析，画像<br>
		坏处是：训练需要大量数据，而且需要每个用户都提供大量数据，每条数据经常并不完整			

	Chatbots 更复杂一点
	
- Ethics in ML 伦理道德

	主要强调样本是否合理，模型训练逻辑是否经过审慎思考。如果用了不合理的样本或训练方法，这个模型可能会产生很多后续坏影响，在事事开源，人人抄袭的今天，很难估计影响范围
	
- python for ML
	- coding tools

		讲解不同工具（代码，开发环境，运行设备）的选择，优缺点
		
		R与python目前同样流行
		
		Python notebooks（.ipynb）可以插入markdown文字，图像等，更适合ML，DA
		
		Python script（.py）更适合程序开发。
		
		IDE：integrated development environment
	
	- [python packages](https://www.coursera.org/learn/the-nuts-and-bolts-of-machine-learning/supplement/qAKAL/python-libraries-and-packages "reading materials")
		- operational packages：pandas numpy syipy 
		- visualization packages
			
			Matplotlib: basic visualization
			
			Seaborn: statistical visualization
			
			Plotly: presentation or publications 甚至以互动
			
		- ML packages: Scikit learn
		
- [ML resources](https://www.coursera.org/learn/the-nuts-and-bolts-of-machine-learning/supplement/eWVHw/find-solutions-online "reading materials")
	- package documentation 查看各个库的使用说明文档
	- stack overflow: coding issue
	- kaggle:
	

## M2 Workflow for building complex models 

## M3 Unsupervised learning techniques
## M4 Tree-based supervised learning
