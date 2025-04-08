# Machine Learning 机器学习

[Course link](https://www.coursera.org/learn/the-nuts-and-bolts-of-machine-learning/home/module/1 "") 

## M1 The different types of machine learning

ML:设定好判断标准，然后让机器自己找到规律。而不是通过精确编程来实现所有的计算。

### Types of ML 机器学习种类
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
	
### Categorical VS Continuous 分类vs连续
	
data and models need to match each other
- Continuous

	Linear regression / Decision Tree regression / Lasso regression / Ridge regression / Polynomial regression / Random Forest regression / Boosting regression

- Categorical/Discrete

	Logistic regression / Navie Bayes classfication / Decision Tree classfication / Random Forest classfication / Boosting classfication
		
### ML in everyday life 日常生活中的机器学习

- Recommendation systems

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

- Chatbots 更复杂一点
	
### Ethics in ML 伦理道德

主要强调样本是否合理，模型训练逻辑是否经过审慎思考。如果用了不合理的样本或训练方法，这个模型可能会产生很多后续坏影响，在事事开源，人人抄袭的今天，很难估计影响范围
	
### python for ML
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
		
### [ML resources](https://www.coursera.org/learn/the-nuts-and-bolts-of-machine-learning/supplement/eWVHw/find-solutions-online "reading materials")
- package documentation 查看各个库的使用说明文档
- stack overflow: coding issue
- kaggle:
	

## M2 Workflow for building complex models 
### P 计划 Plan
	
要分析什么事情，使用对象是谁，项目成果应该是什么样的，数据哪里来，质量如何。<br>
要不要建模，预测对象是连续的还是离散的，用什么回归方法，用什么模型<br>
可用的，适合的分析工具，软件，库，硬件，配合的人

### A 分析 analyze

确定要预测是什么，以及什么样的样本数据结构可以支持接下来的建模分析步骤

- Feature engineering 解决数据组成structure问题
	
	选择有用的属性featrue，转化不合适的数据属性，合并生成一种新属性<br>
	[Selection / Transformation / Extraction](https://www.coursera.org/learn/the-nuts-and-bolts-of-machine-learning/supplement/7Q7BT/explore-feature-engineering "reading materials 常见处理方法")

- solve issues from imbalanced datasets 不均衡的数据集
	
	特别是在处理分类变量时，如果某个分类占比特别高，如超过90%，某种分类占比特别低，如不足10%<br>
	如果所建立的模型需要对占比很低的分类同样敏感，或是对这个分类进行预测<br>
	那么可能需要人为干预样本中分类的比例<br>
	
	去掉多的 Downsampling：当样本数量很多时，如大于10万，直接随机拿掉一部分majority组的样本
	
	或是增加少的 Upsampling：当样本数量不够多时，直接复制minority组的数据；或者用少数组的数据生成新数据，比如每两个的平均值
	
	这么做的影响是让少数组被过度重视，比如在评估每个分类出现概率时，得到的结论就可能是错的，如果只是预测分类则可能还好。总之这是一个应该放在最后才被考虑的调整手段，一定要先尝试其他方法。

```python
# 导入packages
import numpy as np
import pandas as pd
# 导入数据
file_location = ".../*/*.cvs"
df_original = pd.read_cvs(file_location)
# 预览数据表
df_original.head()
# 查看数据结构 print high-level info about data
df_original.info()
# 选择有用属性/去掉无用属性 feature selection
churn_df = df_original.drop(['col_x','col_y','col_z'], axis=1)
# 构建新属性 extraction
churn_df = ['new_col'] = churn_df['col_x'] / churn_df['col_y'] 
# 转换属性 transformation
churn_df['col_need_to_be_transformed'].unique()
churn_df = pd.get_dummies(churn_df, drop_first=True)
```

### C 建模 Construct
- Naive Bayes-GaussianNB
	以朴素贝叶斯算法为例，首先要选一个合适的算法，每个算法里还有不同的模型，这里选用高斯。每个模型适合做什么运算，有什么假设前提是需要大量时间学习和了解的。不过贝叶斯算法对CPU的消耗十分少，虽然过于简单，但仍然很受青睐
	
	然后检查一下分类的分布是不是够平均，不要有低于1：9这种极端不均衡的分布，如果有要进行少数样本放大步骤。然后把数据里不需要的列去掉。
	
	把样本分成训练和测试两组。注意要把少数组也按比例分到这两组里。stratify=y
	
	拟合模型，用测试组测试，跑评估矩阵，看拟合度指标accuracy/precision/recall/f1（参考[Regression analysis](/regressionAnalysis.md "")课程），然后根据结果进行适当的调整。再拟合模型……直到
		
### E 执行 Execute

根据评估结果，思考如何调整模型并得到更好的结果
	
## M3 Unsupervised learning techniques

### K-means

unsupervised partitioning algorithm 划分算法<br>
organize unlabeled data into groups or cluster 将无标签数据归类

- 4 steps to build a model
	1. choose number of centroids(K) and place them in the data space
		
		中心点有多少个如何选择？如果知道要分析的变量有多少类特性，那么就用这个数字。如果没有准确信息，则需要根据其他相关知识猜测一个合理的分类区间，如果什么信息都没有，那就只能随机选取多种数字，再从推算的分类结果来考量K的选择是否合理。
		
	2. assign each datapoint to its nearest centroid
	
	3. recaculate the centroid of each cluster
	
	4. repeat step 2 and 3 until the algorithm converges

- how to avoid poor clustering caused by local minima

	not having an appropriate distance between clusters
	
	- 多试验几个不同的初始controids

	- K-means++ in sikit-learn
	
- K-means do not allow unassigned outliers 

	虽然我们经常把Kmeans叫成cluster聚类算法，但严格说它是partitioning划分算法

	cluster algorithm Vs. partitioning algorithm: 前者允许unassigned outliers

- K-means for color compression
	
	一个用K-mean处理图像-减低颜色数量-的案例

- K-means的优缺点
	
	在数据分布趋近圆形且比较均匀时表现较好，但是不能很好地反映密集程度，最后选择的中心点不一定是数据最密集的地方。
	
	- 如果想找到最密集点，可以用DBSCAN方法
	- 或者用Agglomerative clustering，但这种方法必须规定K值或者cluster间距，否则会迭代到每个点一个分类才能停下。
	- [Scikit-learn内建的聚类算法们](https://www.coursera.org/learn/the-nuts-and-bolts-of-machine-learning/supplement/yynh1/clustering-beyond-k-means "reading materials 页面最后")

### K-means Evaluation Metrics

- Inertia

	Inertia = ∑(x<sub>i</sub>-C<sub>k</sub>)<sup>2</sup>
	
	Inertia值越高，每组数据分布得越稀松。<br>
	Inertia不能告诉我们聚类之间的距离。<br>
	一般K值越高inertia值越低，但是K值增加到一定值以后，inertia不再快速降低。

- Elbow M1ethod 
	
	用不同的K值跑模型，得到inertia，<br>
	画K与Inertia的曲线plot，找到K值的增加不再有显著效果的拐点

- Silhouette Score

	a = the mean distance between the instance and each other instance in the same cluster<br>
	b = the mean distance from the instance to each instance in the nearest other cluster (i.e., excluding the cluster that the instance is assigned to)<br>
	max(a,b) = whichever value is greater
	
	Silhouette coefficient = max(a,b)/(b−a)
​
	[-1，1]：越接近1说明观测点离自己的cluster成员越近，接近0，说明处于两个cluster中间，接近-1说明离自己的成员远而离另一个cluster更近。
	
	silhouette score = mean(Silhouette coefficient<sub>i</sub>)
	
- Silhouette analysis
	
	用不同的K值跑模型，得到silhouette score，<br>
	画silhouette score与K的曲线plot，找到silhouette score最高的K值

- Apply inertia and silhouette score with Python

	略

## M4 Tree-based supervised learning

### Tree-based modeling 决策树模型

### Tune tree-based models 调试决策树

### Bagging, Bootstrap aggregation 引导聚类算法/装袋算法

### Boosting 提升方法
