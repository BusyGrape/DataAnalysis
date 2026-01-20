[Spark, Hadoop, and Snowflake for Data Engineering](https://www.coursera.org/learn/spark-hadoop-snowflake-data-engineering/home/welcome "")

Coursera, by Duke university

当数据的数量多到一定程度，存储无法由一台电脑完成，运算数据远大于内存容量。
如何处理这种规模的数据？
和以往所有事情都可以在一台计算机里完成（不论这个计算机有多大内存多大计算带宽），
硬件调用，数据库结构，数据库文件结构，对数据进行计算处理等
各种逻辑结构都不再一样。

大数据的概念和与之相关的各类新工具都应运而生。

# 几个著名的big data 平台/服务商 
- Apache Hadoop, which is both a specific implementation of the MapReduce algorithm and also a platform for Big Data processing, including distributed storage, as well as other tools.
- Spark, which is a data processing engine that's very popular now for handling big data problems. 
- Snowflake, which began as a distributed storage solution, but it's evolving to also include data processing solutions
- Databricks, which began more on the processing side, but also involves storage. 


# Apache Hadoop
## Hadoop
- Hadoop指，把原始的大数据，进行处理，创建中介/中转数据，然后进行计算并输出结果的过程
- Hadoop同时也指，整套应用MapReduce程序处理数据的生态系统
- 核心是MapReduce算法/语言/程序，和HDFS文件存储系统

## Apache Hadoop
- 大数据平台，可扩展存储
- 应用MapReduce程序/语言的环境（实际上是将数据复制处理，原数据安全，且容错高）
- 分批处理数据（所以不适合对时时更新/响应要求较高的工作）
- 分发工作给多个机器（硬件相同）同步处理工作 (技术门槛高)

  这些机器被称为一个聚落cluster里的节点node
- 一整套生态系统，Hive用于查询，Pig处理数据

## Hadoop MapReduce
programming model
- Map:把文件切割成小文件，然后做一些处理，称之为map phrase，  
- Intermedia files: 再做一次处理，在硬盘上创建足够小的可以被读进内存的中间数据，
- Reduce:把所有中间数据的处理结果合并

## HDFS (Hadoop Distributed File System)
[知乎一篇不错的介绍文章](https://zhuanlan.zhihu.com/p/350080676 "")

HDFS由三部分组成
- NameNode 主 记录文件名称权限等信息，记录被切分的数据块的地址信息
- DataNode 从 每个数据块等大，存放实际数据，每个数据块可以被多次备份
- SecondaryNamenode 备份 NameNode的备份

HDFS 是一套把数据切分并分散到不同硬件上存放，并自带备份策略的文件管理系统。
它包含如何分配和管理数据的物理存放，如何备份数据，如何恢复丢失数据等一系列工作


# Spark
Aparche Spark
## big data engine or platform
- 没有自己的CRM cluster management system 
- 没有自己的distributed data store
- 有自己的driver，决定工作如何被执行，大多数时候是并行计算
- 核心是Resilient Distributed Dataset or RDD
- 是关于数据处理工作如何在内存里被计算的引擎in-memory processing engine

## RDD
我理解RDD是把数据库处理工作分成可以同时进行的几个子工作的这样一套机制。并且这时要处理的数据全部放在内存里，这样接下来的数据交互速度最快。

传统的数据库操作是一条一条语句执行，每条语句都难免要遍历整个数据库。单线程的工作。数据读取受限于硬盘IO，如果是大型数据存在网络里的不同机器上，读取速度主要受限于网络带宽。

至于如何拆分工作，好像是论文机密？

- Transformation 把数据分成几块可以同时处理的部分
- Action 对数据进行计算等处理
- Shuffle 根据需要把数据在已有的分块中挪动，方便后续计算步骤
- Stage 分配，计算组成一个段独立的工作，每个阶段有自己的RDD分配。但是工作阶段之间是有联系的，并且必要的话，可以从上一段工作恢复数据。

## PySpark
python library for Aparche Spark

## Spark SQL
可以对已有数据结构的数据进行操作，类似于panda。
RDD本身是没有dataframe的。


# Snowflake
在云端存储和计算大数据的平台服务。
云可以是现有任意一个云平台AWS/Google Cloud/MS Azure。

我理解它可以提供基于类似Hadoop和/或Spark两种大数据存储和计算方案。但是不需要用户特别了解这两套大数据服务的部署。

Snowflake的收费是基于warehouse的使用（大概是消耗的算力）。

## Big Data Storage
- Shared Everything
- Shared Disk
- Shared Nothing

Snowflake混合用shared disk和shared nothing

## Database Storage
存在Snowflake的data，文件存储结构用户不可知，由snowflake来决定如何分散、备份和优化。
>感觉操作Snowflake就像是在操作一个没有成型的前台界面和固定工作流程的数据库软件部署。包括建立数据库结构，使用者权限，数据抓取汇总公式。。。

## Query Processing
用户只能通过sql语句访问自己的data。
- vitrual warehouses, is an MPP compute cluster
- Snowflake SQL

## Cloud Services
由snowflake负责提供所有管理服务：安全性，权限管理，query优化等。
只能在公共云平台，不能是私有云。

网页界面，编程界面，提供ODBC/JDBC接口，python/spark等程序接口，分析工具直连接口。

## Snowsight
Snowsight is a tool built on top of Snowflake to facilitate data exploration, visualization, and collaboration


# Databricks
用spark处理数据运算，用Azure建立数据工作流程的平台。
主要针对大数据处理，和机器学习，有专门服务ML开发实验的工具包，组件，界面。
可以理解是想模仿GitHub打造一个ML的开源共享开发模型的工具加平台加社区的怪物。而且是付费加入免费开源互助服务队伍。

## Spark Notebooks
内置Databrick的编辑器，类似于Jupyter吧，或者就是。

## Data
介绍云端和本地数据如何使用

## Workspace
介绍操作界面，用户，权限，项目管理

## PySpark
和普通panda比，大数据效率提升。
还可以import pyspark.pandas as ps，就可以用panda的fn/api写code，
但是会自动对大数据调优。因为实际上存储数据的方式是按照spark的结构

## DBFS
The Databricks File System (DBFS) is a distributed file system designed for cloud-based data analytics within the Databricks platform. 
It provides a central and scalable storage layer that allows data engineers and data scientists to seamlessly manage and access data in their data lake or cloud storage. 
DBFS is agnostic to the underlying cloud storage (e.g., AWS S3, Azure Data Lake Storage, Google Cloud Storage) and provides a unified way to interact with data.

估计也是一种分布式存储，MS官方一堆使用指导，推荐和不推荐

## 演示为ML预设的一套工作流程界面
可以分别记录跑的什么model结果什么，不同批次test结果是什么等等

可以给model存不同版本随时调用

## Serveless Compute

## MLops
想做一个大家分别开发，然后能把好的东西汇总的生态系统，开源或者共享开发平台，参照DevOps平台。

## 介绍怎么在DB平台上用别人开发过的模型

好么最后就是想让大家换个话题继续免费开源，互相帮助，用爱发电

