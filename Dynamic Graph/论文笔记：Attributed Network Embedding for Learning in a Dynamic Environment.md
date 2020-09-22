# 论文笔记：Attributed Network Embedding for Learning in a Dynamic Environment[DANE]

## 1.方法模型

key points: 把embedding转换为广义特征分解问题，**网络拓扑与节点特征相结合**的动态网络表征学习

### 1.Task&Problem Formulation

problem of dynamic attributed network embedding: initiate an offline model at the very beginning, based on which an online model is presented to maintain the freshness of the end attributed network embedding results

nodes: $\mathcal{U}^{(t)}=\left\{u_{1}, u_{2}, \ldots, u_{n}\right\}$, each with d attributes,  $X^t$ represents attributes 

two problem:

- static:  offline model of DANE at time step t : given network topology A(t ) and node attributes X(t ); output attributed network embedding Y(t ) for all nodes.

- dynamic:  onlinemodel of DANE at time step t+1: given network topology A(t+1) and node attributes X(t+1), and intermediate embedding results at time step t ; output attributed network embedding
  Y(t+1) for all nodes.

**动态属性网络（Dynamic Attributed Networks）：**网络结构或节点属性都会随时间改变的网络

- 提出了一种**离线嵌入方法**作为基本模型，从**网络结构和节点属性两个方面**保留节点的近似度，以实现**共识嵌入表示**
- 为了在**网络结构和节点属性发生改变**时及时获得更新的嵌入表示，提出了一种**在线模型**，基于**矩阵扰动理论更新共识嵌入**。

### 2. Models

![img](https://img-blog.csdnimg.cn/20191115195545295.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTg3Mzg3MQ==,size_16,color_FFFFFF,t_70)

#### DANE: Offline Model

$A^{(t)}$ attributed matrix at time t,  $D_A$ be the diagonal matrix with $D_A^{(t)}(i,i)=\sum_{j=1}^{n}A^{(t)}(i,j) $

$\mathrm{L}_{\mathrm{A}}^{(t)}=\mathrm{D}_{\mathrm{A}}^{(t)}-\mathrm{A}^{(t)}$ is a Laplacian matrix.

minimize the loss: $\frac{1}{2} \sum_{i, j} \mathrm{A}^{(t)}(i, j)|| \mathrm{y}_{i}-\mathrm{y}_{j} \|_{2}^{2}$

It ensures that connected nodes are close to each other in the embedded space

可以把问题转化为特征值问题，$\mathrm{L}_{\mathrm{A}}^{(t)} \mathrm{a}=\lambda \mathrm{D}_{\mathrm{A}}^{(t)} \mathrm{a}$

1是特征值0对应的唯一的特征向量，再选从$a_2$开始的top-k特征向量

对$A^{(t)}$这样操作后得到$Y_A^{(t)}$= [a2, ..., ak , ak+1].

使用同样方法操作W(t), we first normalize attributes of each node and obtain the cosine similarity matrix W(t). Afterwards, we obtain the top-k eigenvectors $Y_X^{(t)}$= [b2, ..., bk+1] of the generalized eigen-problem corresponding to W(t ).

用two intermediate embeddings$Y_A^{(t)}$和$Y_X^{(t)}$来解决noise的问题，然鹅这两者之间可能是独立的，尝试maximize their correlations（or equivalently minimize their disagreements）【为了学习共识向量】

寻找两个projection vectors $p_A^{(t)}$和$p_X^{(t)}$, 使得correlation is maximized after projection.

$$\begin{aligned}
&\max _{p_{A}^{(t)}, p_{X}^{(t)}} \mathbf{p}_{A}^{(t)^{\prime}} \mathbf{Y}_{A}^{(t)^{\prime}} \mathbf{Y}_{\mathbf{A}}^{(t)} \mathbf{p}_{\mathbf{A}}^{(t)}+\mathbf{p}_{\mathbf{A}}^{(t)^{\prime}} \mathbf{Y}_{\mathbf{A}}^{(t)^{\prime}} \mathbf{Y}_{\mathbf{X}}^{(t)} \mathbf{p}_{\mathbf{X}}^{(t)}\\
&+p_{X}^{(t)^{\prime}} Y_{X}^{(t)^{\prime}} Y_{A}^{(t)} P_{A}^{(t)}+p_{X}^{(t)^{\prime}} Y_{X}^{(t)^{\prime}} Y_{X}^{(t)} p_{X}^{(t)}\\
&\text { s.t. } \quad \mathrm{p}_{\mathrm{A}}^{(t)^{\prime}} \mathrm{Y}_{\mathrm{A}}^{(t)^{\gamma}} \mathrm{Y}_{\mathrm{A}}^{(t)} \mathrm{p}_{\mathrm{A}}^{(t)}+\mathrm{p}_{\mathrm{X}}^{(t)^{\prime}} \mathrm{Y}_{\mathrm{X}}^{(t)^{\prime}} \mathrm{Y}_{\mathrm{X}}^{(t)} \mathrm{p}_{\mathrm{X}}^{(t)}=1
\end{aligned}$$



 $\gamma$ be the Lagrange multiplier for the constraint, 拉格朗日乘数法，求导求极值，等同于求下面特征问题的特征值：

<img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200918210656698.png" alt="image-20200918210656698" style="zoom:80%;" />

保留top-l eigenvectors，the final consensus embedding representation can be computed：

$\mathrm{Y}^{(t)}=\left[\mathrm{Y}_{\mathrm{A}}^{(t)}, \mathrm{Y}_{\mathrm{X}}^{(t)}\right] \times \mathrm{P}^{(t)}$

#### Online Model of DANE

大多数现实世界中的网络，属性网络也不例外，通常在两个连续时间步之间的时间维上平稳地演化

因此使用$\Delta A$ 和$\Delta X$ to denote the perturbation of network structure and node attributes between two consecutive time steps t and t +1,(编码网络结构和属性的扰动)

With these, the diagonal matrix and Laplacian matrix envolve:

$$\begin{array}{l}
\mathrm{D}_{\mathrm{A}}^{(t+1)}=\mathrm{D}_{\mathrm{A}}^{(t)}+\Delta \mathrm{D}_{\mathrm{A}}, \quad \mathrm{L}_{\mathrm{A}}^{(t+1)}=\mathrm{L}_{\mathrm{A}}^{(t)}+\Delta \mathrm{L}_{\mathrm{A}} \\
\mathrm{D}_{\mathrm{X}}^{(t+1)}=\mathrm{D}_{\mathrm{X}}^{(t)}+\Delta \mathrm{D}_{\mathrm{X}}, \quad \mathrm{L}_{\mathrm{X}}^{(t+1)}=\mathrm{L}_{\mathrm{X}}^{(t)}+\Delta \mathrm{L}_{\mathrm{X}}
\end{array}$$

如前一小节所讨论的，属性网络嵌入离线设置的问题归结为求解广义特征值问题。尤其是离线模型，着重于寻找广义特征值问题的最小特征值对应的上特征向量。因此，实现嵌入在线更新的核心思想是开发一种有效的方法来更新top特征向量和特征值。否则，每个时间步都要进行广义特征分解，时间复杂度高，不实用

根据矩阵摄动理论[39]，在新的时间步长嵌入网络结构时，我们得到以下方程：

$\left(\mathrm{L}_{\mathrm{A}}^{(t)}+\Delta \mathrm{L}_{\mathrm{A}}\right)(\mathrm{a}+\Delta \mathrm{a})=(\lambda+\Delta \lambda)\left(\mathrm{D}_{\mathrm{A}}^{(t)}+\Delta \mathrm{D}_{\mathrm{A}}\right)(\mathrm{a}+\Delta \mathrm{a})$

对于每个具体的（特征值-特征向量）pair，有以下方程：

$$\left(\mathrm{L}_{\mathrm{A}}^{(t)}+\Delta \mathrm{L}_{\mathrm{A}}\right)\left(\mathrm{a}_{i}+\Delta \mathrm{a}_{i}\right)=\left(\lambda_{i}+\Delta \lambda_{i}\right)\left(\mathrm{D}_{\mathrm{A}}^{(t)}+\Delta \mathrm{D}_{\mathrm{A}}\right)\left(\mathrm{a}_{i}+\Delta \mathrm{a}_{i}\right)$$

- A.Computing the change of eigenvalue

  展开上面的式子，有<img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200918212712487.png" alt="image-20200918212712487" style="zoom:80%;" />

  <img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200918212836095.png" alt="image-20200918212836095" style="zoom:80%;" />

- B.Computing the change of eigenvector

  <img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200918213045342.png" alt="image-20200918213045342" style="zoom:67%;" />

  <img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200918213405548.png" alt="image-20200918213405548" style="zoom:80%;" />

  <img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200918213510555.png" alt="image-20200918213510555" style="zoom:80%;" />

  

- C.Computational Complexity Analysis

### 3.Experiments









## 2. 写作

proximity

manifested 清楚的，显现的

sth could advance xxx tasks

Nonetheless

fading 衰退

of fundamental importance

necessitates 使。。。成为必要

consensus 共识

accordingly 相应地

ubiquitous 无处不在的

be affiliated with

predominately

inevitably

daunting 令人生畏的

be of paramount importance 最重要的

synthetic 合成的

mitigating 减轻缓和

jeopardized 冒险的，有害的

mitigate 减轻 缓和

Hense

boils down 归结为

perturbation 扰动

likewise 类似地