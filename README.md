# Polyregression_simple_NN_pratice
poly-regression implemented by neural network

## Tensorflow 实现多项式回归， L1，L2正则化
### 程序基本思想：
#### step1 输入向量的预处理
将输入的向量例如
\begin{equation} x = \begin{bmatrix} x_1 \\ x_2 \\ ...\\x_n \end{bmatrix} \end{equation}
进行扩维，变成
\begin{equation} feature = \begin{bmatrix} {x_1}^0 & {x_1}^1 & {x_1}^2 & {x_1}^3\\ {x_2}^0 & {x_2}^1 & {x_2}^2 & {x_2}^3\\ ...\\{x_n}^0 & {x_n}^1 & {x_n}^2 & {x_n}^3\end{bmatrix} \end{equation}
#### step2 构建神经网络结构
构造简单神经网络(由于上面已经对特征进行了扩充，所以不需要中间隐藏层即可完成模型的构建)，进行训练。这里为了避免过拟合的情况，使用了L2正则（ridge回归）。当然也可以根据需要构建lasso回归和Elastic Network。只需对代码做少量修改。为了方便，我们这里只使用了L2正则。
#### step3 画出得到的回归曲线
