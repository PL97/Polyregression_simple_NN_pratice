
# coding: utf-8

# ## Tensorflow 实现多项式回归， L1，L2正则化
# ### 程序基本思想：
# #### step1 输入向量的预处理
# 将输入的向量例如
# \begin{equation} x = \begin{bmatrix} x_1 \\ x_2 \\ ...\\x_n \end{bmatrix} \end{equation}
# 进行扩维，变成
# \begin{equation} feature = \begin{bmatrix} {x_1}^0 & {x_1}^1 & {x_1}^2 & {x_1}^3\\ {x_2}^0 & {x_2}^1 & {x_2}^2 & {x_2}^3\\ ...\\{x_n}^0 & {x_n}^1 & {x_n}^2 & {x_n}^3\end{bmatrix} \end{equation}
# #### step2 构建神经网络结构
# 构造简单神经网络(由于上面已经对特征进行了扩充，所以不需要中间隐藏层即可完成模型的构建)，进行训练。这里为了避免过拟合的情况，使用了L2正则（ridge回归）。当然也可以根据需要构建lasso回归和Elastic Network。只需对代码做少量修改。为了方便，我们这里只使用了L2正则。
# #### step3 画出得到的回归曲线
# ---------------------------------------------------------------------------------------------------------------------------------

# 引入必要的包，这里需要的是numpy，tensorflow， 画图工具包matplotlib， 和math

# In[1]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math


# 手动创建数据,这里创建函数 $y = 10x^2 + 3$，并引入部分噪声。最后将创建数据进行显示

# In[2]:


def f(x):
#     return 10 * np.sin(x)
    return 10 * x**2 + 3
data_size = 100
D = 4
x = np.random.uniform(low = 0, high = 10, size = data_size)
feature = np.array([[i**j for i in x] for j in range(D)]).transpose()
noise = np.random.normal(0, 25, data_size)
label = np.mat(f(x) + noise).transpose()
print(label.shape)

plt.figure(figsize = (8, 8))
plt.scatter(x, list(label))
plt.show()


# 创建多项式回归模型

# In[3]:


def PolyRegression_NN(feature, label, lamda = 15, learning_rate = 0.1, batch_size = 100):
    X = tf.placeholder(tf.float32, [None, D], name='X')
    Y = tf.placeholder(tf.float32, [None, 1], name='Y')
    W = tf.Variable(tf.random_normal([D, 1]),name = 'weight')
    Y_pred = tf.matmul(X,W)
    
    loss = tf.log(tf.add(tf.reduce_mean(tf.square(Y - Y_pred)),  lamda*tf.norm(W, ord = 2)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    # 使用全局参数初始化器
    sess = tf.Session()
    tf.global_variables_initializer().run(session = sess)
    pre_w = sess.run(W)
    for i in range(10000):
        rand_index = np.random.choice(x.shape[0], size=batch_size)
        batch_xs, batch_ys = feature[rand_index, :], label[rand_index]
        optimizer.run({X: batch_xs, Y: batch_ys}, session=sess)
        # 计算权重矩阵的最近两次的偏差
        curr_w = sess.run(W)
        deviation = curr_w - pre_w
        move = np.dot(deviation.transpose(), deviation)
        # 迭代终止条件——两次的权重矩阵的偏差小于一定值
        if move[0][0] < 0.0001:
            print('break at {}th iteration'.format(i))
            break
        pre_w = curr_w
    return sess.run(W)


# In[4]:


W = PolyRegression_NN(feature, label)


# 图形化展示结果

# In[5]:


# 将拟合效果图形化展示
def new_f(w, X):
    x_tran = np.array([[x**i for x in X] for i in range(len(w))]).transpose()
    return np.dot(x_tran, w)
new_x = [i for i in range(math.floor(min(x)), math.ceil(max(x)), 1)]
plt.plot(new_x, new_f(W, new_x))
plt.scatter(x, list(label), c = 'r')
plt.show()

