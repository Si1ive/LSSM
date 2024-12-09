# 模型迭代
## 默认配置
64 128 256 512
lr 1e-4

## Version1
F1 0.8515
batchsize 12 
6.89M  1.65G
34.6M  7.57G
1. 训练到49轮左右时lr迭代到了1e-5，得分提升了
2. 又开了新一轮，直接把lr设定为1e-5，并且把迭代频率考察epoch数从15换成了10，得分再也提不动了
3. 疑似10轮太少了，导致学习率拼命减，减过头更新不动了

## Version2
1. 将dim从64改成96开始
参数量直接从37 干到77
训练速度从1.5s左右干到15s左右
从不到十分钟干到仨小时
2. dim改成80
参数量53
训练速度5s 一轮一小时 训练不动没训练

## Version3
6.02M  1.58G
38M    8.37G
dim64 训练速度8s左右,参数38，参数量变化很小，速度慢了很多 1:22
1. 修改所有降通道操作，都改成先降四倍，再提两倍
2. 移除AFF融合四个D的模块
3. 更改为可以融合边缘增强的架构，即解码器中每一层都输出一个结果，然后都计算到损失当中
4. 多种方法：
   1. Laplace对于decoder中每一层数据应该独立卷积然后下采样还是就一次卷积得到laplace灰度图
   2. 因为还有差异特征等，所以解码层中肯定是先进行支线主线融合后再进行上采样
5. 修改了损失函数,使用WBCE，要求模型的输出为logits，未使用softmax和sigmoid，并且通道数和GT要一致，为1,也就是修改模型的num_class

## Vesion4
6.01M  1.57G
38.3M  8.37G
1024 - 'f_score=0.892704126076433'
没多高，把batsize设置为大点，
1.  解码层的BN，全改成LN,因为批数12属于很小，100多的才算大，对于这么小的批数，BN适得其反
2.  卷积偏置项
3.  开头的ABconv修改
### 遭遇bug
预测值全变成了0
解决：清梯度，损失函数有问题，修改了损失函数，增加了清缓存

## Version5
1. 输出文件夹检验
2. batchsize设置为128
3. dim设置为96开始，参数量直接干到85
4. VSSM是通用的，注意模型版本跟VSSM的对应关系
