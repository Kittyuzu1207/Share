# 论文笔记：DynGEM: Deep Embedding Method for Dynamic Graphs

## 1.方法模型

### 1.Task&Problem Formulation

把动态图视作一连串的snapshots: $G=\{G_1,G_2,...G_T\}$, growing graphs: nodes和边可以新增

a dynamic graph embedding: a time-series of mappings $F=\{f_1,f_2,...f_T\}$

想要学习到*stable* 的embedding，【考虑到snapshots之间的关系：如果$G_T$到$G_T+1$ 变化不大，embedding output也应该变化不大

--定义 *absolute stability*：the ratio of the difference between embeddings to that of the difference between adjacency matrices

<img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200921165712929.png" alt="image-20200921165712929" style="zoom:80%;" />

F是子图的embedding，S是子图的邻接矩阵，这个定义depends on the sizes of matrices involved

--定义*relative stability*：invariant to the size of adjacency and embedding matrices

<img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200921170120328.png" alt="image-20200921170120328" style="zoom:80%;" />

--定义*stability constant*:

<img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200921170231426.png" alt="image-20200921170231426" style="zoom:80%;" />

We say that a dynamic embedding F is stable as long as it has a small stability constant. Clearly

### 2.Models

#### Handling growing graphs

当nodes增加时，要如何调整hidden layers， hidden units...提出了一种启发式的a heuristic, PropSize, to compute new layer sizes for all layers which ensures that the sizes of consecutive layers are within a certain factor of each other

**PropSize**

对于encoder，计算layer widths are computed for each pair of consecutive layers($l_k$ and $l_{k+1}$), 直到满足以下condition:

![image-20200921171640566](C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200921171640566.png)

where $0<\rho<1$ ,是一个hyperparameter, 对于任意一个pair如果这个condition都不满足，the layer width for $l_{k+1}$ is increased to satisfy the heristic（**拓宽$l_{k+1}$ 层**). embedding layer的size是固定不变的，d. 如果这个condition在倒数第二步不满足（即最后一个hidden 层和embedding层组成的pair)，就add more layers 直到满足。(**加深层**)

this procedure is also applied to the decoder layers(从$\hat{x}$ 到embedding layer $\hat{y}$)

在确定好hidden layer和hidden units in each layer之后，adopt *Net2WiderNet* and *Net2DeeperNet* to expand deep autoencoder. 

*Net2WiderNet*: widen layer: add more hidden units

*Net2DeeperNet* : increase new layers between two existing layers

总结：这个section 在widen and deepen autoencoders

**Loss Function and Training**

three objectives: 

$$L_{n e t}=L_{g l o b}+\alpha L_{l o c}+\nu_{1} L_{1}+\nu_{2} L_{2}$$

$\alpha$,${v_1}$,$v_2$,控制不同loss的权重

$L_{loc}=\sum_{i, j}^{n} s_{i j}\left\|y_{i}-y_{j}\right\|_{2}^{2}$: the **first-order** proximity which corresponds to local structure of the graph【依赖y，有监督】

$L_{glob}=\|(\hat{X}-X) \odot B\|_{F}^{2}$: the **second-order** proximity which corresponds to global neighborhood of each node in the graph and is preserved by an **unsupervised reconstruction** of the neighborhood of each node. 【依赖x，无监督】

$\mathbf{b_i}$ is a vector with $b_{ij}=1$ if $s_{ij}=0$ else $b_{ij}=\beta>1$. 【错误重构的惩罚比加了unobserved edges的惩罚更大】

L1,L2：权重矩阵的范式，防止过拟合

**Stability by reusing previous step embedding**

对于第一步，随机初始化parameters $\theta_1$, 对于之后的每一步，都把上一步的parameter作为当前步的初始parameter，Thisr results in direct knowledge transfer of structure from $f_{t-1}$ to $f_t$, so the model only needs to learn about changes between $G_{t-1}$ and $G_t$. 也加速了收敛

**Techniques for scalability**

We use ReLU in all autoencoder layers to support weighted graphs since ReLU can construct arbitrary positive entries of $s_i$.

用nesterov momentum进行优化

### 3. Experiments&Analysis

**Graph construction & Link Prediction**

<img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200922134010359.png" alt="image-20200922134010359" style="zoom:67%;" />

line prediction: randomly hide 15% of the network edges at time t, 

<img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200922134803510.png" alt="image-20200922134803510" style="zoom:67%;" />

颜色表明community，改变一部分node的community，看embedding 的稳定性

<img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200922135122857.png" alt="image-20200922135122857" style="zoom:80%;" />

**Application to Anomaly Detection** 异常检测

看nodes之间的communication水平

**Effect of Layer Expansion**

加了layers之后MAP提升了



## 2.写作

wide application

constantly

unsatisfactory

in terms of stability, flexibility and efficiency

recent advances

demonstrate

smoothness

heuristic 启发式的

penultimate 倒数第二的