# Neural Network


### 为什么必须把前后层的每一个神经元与所有其他层的神经元相互连接

①这种一致的完全连接形式事实上可以相对容易地编码成计算机指令

②神经网络的学习过程将会弱化这些实际上不需要的连接(也就是这些连接的权重将趋近于0)

![神经元](https://github.com/peanwang/Neural-network/blob/master/%E7%AC%94%E8%AE%B0/%E7%A5%9E%E7%BB%8F%E5%85%83.PNG)

![神经网络](https://github.com/peanwang/Neural-network/blob/master/%E7%AC%94%E8%AE%B0/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C.PNG)


### 矩阵

![矩阵](https://github.com/peanwang/Neural-network/blob/master/%E7%AC%94%E8%AE%B0/%E7%9F%A9%E9%98%B5.png)


### 反向传播(Back Propagation)

![bp](https://github.com/peanwang/Neural-network/blob/master/%E7%AC%94%E8%AE%B0/bp.png)

## 更新权重 & 梯度下降

![梯度下降](https://github.com/peanwang/Neural-network/blob/master/%E7%AC%94%E8%AE%B0/gradient_descent.PNG)

误差反向传播到网络的每一层后，应该使用误差来知道调整链接权重。
但是这个节点都不是简单的线性分类器。这些节点都是加权后的信号进行求和，并应用了sigmoid函数，将所得到的结果输出给下一层的节点。因此，如何真正地更新链接这些复杂节点链接的权重呢？

![公式](https://github.com/peanwang/Neural-network/blob/master/%E7%AC%94%E8%AE%B0/%E5%85%AC%E5%BC%8F.PNG)


即梯度下降算法 Gradient descent
梯度下降找到代价函数的最小值

|代价函数| costfunction
|--|--
|①| $(目标值-实际值)$
|②| $\|目标值-实际值\|$ 
|③| $(目标值-实际值)^2$


第一个代价函数 会由于正负误差相互抵消。所以淘汰
第二个代价函数，这会是斜率在最小值附件不是连续的。所以淘汰。
第三个的好处：
1. 很容易计算出梯度下降的斜率
2. 代价函数平滑连续。会使梯度下降发挥很好的作用
3. 越接近最小值，梯度越小。这会使这个函数超调的风险很低。

![梯度推导](https://github.com/peanwang/Neural-network/blob/master/%E7%AC%94%E8%AE%B0/gradient_descent2.jpg)



### 瞎测试

性能和学习率的关系：

![学习率](https://github.com/peanwang/Neural-network/blob/master/%E7%AC%94%E8%AE%B0/Figure_1.png)


性能和隐藏层神经元数量的关系：

![隐藏层神经元数量](https://github.com/peanwang/Neural-network/blob/master/%E7%AC%94%E8%AE%B0/Figure_2.png)


性能和世代的关系：

![世代](https://github.com/peanwang/Neural-network/blob/master/%E7%AC%94%E8%AE%B0/Figure_3.png)