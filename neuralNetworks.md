TensorFlow is a library for developing and training machine learning models. Keras is an API built on top of TensorFlow designed for neural networks and deep learning.

# Intro to Deep Learning
Objectives:
- create a fully-connected neural network architecture
- apply neural nets to two classic ML problems: regression and classification
- train neural nets with stochastic gradient descent, and
- improve performance with dropout, batch normalization, and other techniques

## A Single Neuron
一个神经元：一个线性模型
```mermaid
flowchart LR
    A(("X")) -- w --> B(("\+"))
    B --> C["y"]
    D(("1")) -- b --> B
    C@{ shape: text}
```

## Deep Neural Networks
### Layers
层：一组线性模型/一组神经元，每个模型/神经元的输入都是一样的。
```mermaid
flowchart LR
    X0(("X0")) --> D0(("\+"))
    X1(("X1")) --> D0
    1(("1")) --> D0
    D0 --> O0["y0"]
    X0 --> D1(("\+"))
    X1 --> D1
    1 --> D1
    D1 --> O1["y1"]
    O0@{ shape: text}
    O1@{ shape: text}
```

不同种类的层可以采用不同的数据拟合方法。理论上说，我们可以采用任何算法/公式来考虑数据之间的关系。

Dense Layer：是线性回归模型层。

> 为什么一层里要有多个神经元？
> 
> [知乎回答一，同一层的不同神经元，w b 是随机初始值，这样每个神经元的作用不会一样](https://www.zhihu.com/question/270100538 "")
>
> 如果只是做线性回归多个神经元没有太多意义，但是做非线性回归，加上激活公式，多个神经元的作用就比较明显

### The Activation Function
给层输出加上一个“调节器”。

ReLU，只输出大于0的结果，这样做，经过两个Dense Layer，可以拟合曲线。
> [知乎一篇激活函数介绍](https://zhuanlan.zhihu.com/p/690650173 "")
> <a href="https://github.com/BusyGrape/DataAnalysis/blob/main/links/简单理解神经网络中常用数学函数——激活函数.mhtml" target="_blank">(右击另存网页)</a>

### Python code
```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    # the hidden ReLU layers
    layers.Dense(units=4, activation='relu', input_shape=[2]),
    layers.Dense(units=3, activation='relu'),
    # the linear output layer 
    layers.Dense(units=1),
])
```

## Stochastic Gradient Descent
### The Loss Function
线性回归，通常用平均误差绝对值法MAE，也可以用其他的比如MSE
> 如何通过loss function调整下一次迭代的系数w b
> 
> [这个帖子的计算过程基本回答清楚了线性回归模型算法工作原理](https://juejin.cn/post/7480369529158746146 "")
> <a href="https://github.com/BusyGrape/DataAnalysis/blob/main/links/梯度计算.mhtml" target="_blank">(右击另存网页)</a>
> 
> [知乎文章，多个神经元，多个输入值，如何对应计算](https://zhuanlan.zhihu.com/p/690647602 "")
> <a href="https://github.com/BusyGrape/DataAnalysis/blob/main/links/简单理解神经网络中常用数学函数——线性函数.mhtml" target="_blank">(右击另存网页)</a>

### Optimizer
优化算法，找到让loss最小的weights。
所有的深度学习DL模型优化算法，都属于SGD家族。是迭代型算法。

每一次迭代的基本工作内容是：
- 从训练数据里，随机选一组样本minibatch，然后用上一次迭代的w做预测
- 计算预测结果与实际结果之间的差异，loss function
- 然后调整系数weights的数值，让loss变小

直到遍历完整个训练数据，一个epoch。

Learning Rate & Batch Size，是SGD的两个对结果影响最大的超级参数，
虽然可以手动做一个超级参数调优方案，
但是一般用Adam算法（一种SGD算法），它里面内置了learning rate自我调整

```python
model.compile(
    optimizer="adam",
    loss="mae",
)
```
### Training code
选择batch size（每次迭代抽多少个样本） 和 epochs（每个样本总共被训练几次）

```python
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,
    epochs=10,
)
```

> loss function是在通过所有层以后才启动计算？
> 
> [知乎文章似乎肯定了这个猜想](https://zhuanlan.zhihu.com/p/683866243 "")
> <a href="https://github.com/BusyGrape/DataAnalysis/blob/main/links/神经网络基础内容.mhtml" target="_blank">(右击另存网页)</a>

## Overfitting and Underfitting

### Capacity
A model's capacity refers to the size and complexity of the patterns it is able to learn. For neural networks, this will largely be determined by how many neurons it has and how they are connected together. 

underfit了，要么加神经元数量，要么加层数

### Early Stopping
为了防止overfit，在构建模型前，先定义叫停标准
The early stopping callback will run after every epoch

```python
from tensorflow import keras
from tensorflow.keras import layers, callbacks

early_stopping = callbacks.EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=20, # how many epochs to wait before stopping
    restore_best_weights=True,
)

# model
# model.complie
```

After defining the callback, add it as an argument in fit (you can have several, so put it in a list). Choose a large number of epochs when using early stopping, more than you'll need.

```python
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,
    epochs=500,
    callbacks=[early_stopping], # put your callbacks in a list
    verbose=0,  # turn off training log
)

# plot the loss for both training and validating data
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot();
print("Minimum validation loss: {}".format(history_df['val_loss'].min()))
```

## Dropout and Batch Normalization
不包含任何神经元的层

### Dropout
随即扔掉一部分神经元，避免过度拟合/过度学习特征/overfit

```python
keras.Sequential([
    # ...
    layers.Dropout(rate=0.3), # apply 30% dropout to the next layer
    layers.Dense(16),
    # ...
])
```

### Batch Normalization
把输入数据按这部分数据的mean / std 进行放缩调整

```python
# 可以放在任意两层之间
layers.Dense(16, activation='relu'),
layers.BatchNormalization(),
Layers.Dense(...)

# 可以放在一层的神经元和其激活公式之间
layers.Dense(16),
layers.BatchNormalization(),
layers.Activation('relu'),

# 如果放在第一层，就跟Sci-Kit Learn's StandardScaler效果一样
```

## Binary Classification
二分类，激活公式和loss function都不一样

### Cross-Entropy
loss function，二分类用binary_crossentropy

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)
```

### Sigmoid Function
给最后一层加 sigmoid

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(4, activation='relu', input_shape=[33]),
    layers.Dense(4, activation='relu'),    
    layers.Dense(1, activation='sigmoid'),
])
```