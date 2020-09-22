# 论文笔记： GraphWaveNet for Deep Spatial-Temporal Graph Modeling

## 1.方法模型

（是时空图，拓扑结构是静态的且只有节点或边缘特征发生变化的图）

### 1.Task &Problem Formulation

本文的idea： By developing a novel adaptive dependency matrix and learn it through node embedding, our model can precisely capture the hidden spatial dependency in the data[我们提出了一个图卷积层，通过端到端的监督训练从数据中学习自适应邻接矩阵。这样，自适应邻接矩阵保留了隐藏的空间相关性。采用堆叠扩张的随机卷积来捕获时间相关性。随着隐层数目的增加，叠加扩张的随机卷积网络的感受野大小呈指数增长]

A basic assumption : a node’s future information is conditioned on its historical information as well as its neighbors’ historical information.

Spatial-Temporal related work

- RNN-based

  使用图卷积filter传递给递归单元的输入和隐藏状态来捕获时空依赖性;使用node-RNN和edge-RNN来处理时间信息的不同方面.基于RNN的方法的主要缺点是对长序列的处理效率低，并且当它们与图卷积网络结合时，其梯度更容易爆炸。

- CNN-based

  基于CNN的方法将图卷积与标准1D卷积相结合

  这两种方法虽然计算效率高，但必须多层叠加或使用全局池来扩展神经网络模型的接受域

**Problem Definition**

$G=(V,E)$ , adjacency matrix $\mathbf{A} \in \mathbf{R}^{N \times N}$, $A_{ij}$ is one or zero

at each time step t, the graph G has a dynamic feature matrix $\mathbf{X}^{(t)} \in \mathbf{R}^{N \times D}$

--Given a graph and  its historical S step graph signals, our problem is to learn a function f which is able to forecast its next T step graph signals.

-->$$\left[\mathbf{X}^{(t-S): t}, G\right] \stackrel{f}{\rightarrow} \mathbf{X}^{(t+1):(t+T)}$$]

### 2. Models

#### Graph Convolution Layer

GCN layer: $\mathrm{Z}=\tilde{\mathbf{A}} \mathrm{X} \mathbf{W}$

Li et al., modeled the diffusion process of graph signals with K finite steps.

$\mathrm{Z}=\sum_{k=0}^{K} \mathbf{P}^{k} \mathbf{X} \mathbf{W}_{\mathbf{k}}$

$P^k$ represents the power series of the transition matrix.

考虑有向图：backward+forward

$$\mathrm{Z}=\sum_{k=0}^{K} \mathrm{P}_{f}^{k} \mathrm{X} \mathbf{W}_{k 1}+\mathbf{P}_{b}^{k} \mathrm{X} \mathbf{W}_{k 2}$$

**Self-adaptive Adjacency Matrix**

This self-adaptive adjacency matrix does not require any prior knowledge and is learned end-to-end through stochastic gradient descent.  We achieve this by randomly initializing two node embedding dictionaries with learnable parameters E1;E2.

$$\tilde{\mathbf{A}}_{a d p}=\operatorname{SoftMax}\left(\operatorname{ReLU}\left(\mathbf{E}_{1} \mathbf{E}_{2}^{T}\right)\right)$$

We name E1 as the source node embedding and E2 as the target node embedding. 

- By multiplying E1 and E2, we derive the spatial dependency weights between the source nodes and the target nodes. 
- ReLU: eliminate weak connections
- Softmax: normalize the self-adaptive adjacency matrix

By combining pre-defined spatial dependencies and self-learned hidden graph dependencies:

$$\mathrm{Z}=\sum_{k=0}^{K} \mathrm{P}_{f}^{k} \mathrm{X} \mathrm{W}_{k 1}+\mathrm{P}_{b}^{k} \mathrm{X} \mathbf{W}_{k 2}+\tilde{\mathbf{A}}_{a p t}^{k} \mathrm{X} \mathbf{W}_{k 3}$$

When the graph structure is unavailable：

$$\mathrm{Z}=\tilde{\mathbf{A}}_{a p t}^{k} \mathrm{X} \mathbf{W}_{k}$$

#### Temporal Convolution Layer

使用**dilated causal convolution**[扩张性因果卷积] to capture a node's temporal trends

dilated casual convolution networks are able to handle longrange sequences properly in a non-recursive manner, which facilitates parallel computation and alleviates the gradient explosion problem.

<img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200916135244150.png" alt="image-20200916135244150" style="zoom:67%;" />

given a 1D sequence input:

**Gated TCN**

$$\mathbf{h}=g\left(\boldsymbol{\Theta}_{1} \star \mathcal{X}+\mathbf{b}\right) \odot \sigma\left(\boldsymbol{\Theta}_{2} \star \mathcal{X}+\mathbf{c}\right)$$

#### Framework of GraphWaveNet

<img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200916135924024.png" alt="image-20200916135924024" style="zoom:80%;" />

A spatial-temporal layer is constructed by a graph convolution layer (GCN) and a gated temporal convolution layer (Gated TCN) which consists of two parallel temporal convolution layers (TCN-a and TCN-b). By stacking multiple spatial-temporal layers, GraphWaveNet is able to handle spatial dependencies at different temporal levels.

use mean absolute error (MAE) as the training objective:

$$L\left(\hat{\mathbf{X}}^{(t+1):(t+T)} ; \Theta\right)=\frac{1}{T N D} \sum_{i=1}^{i=T} \sum_{j=1}^{j=N} \sum_{k=1}^{k=D}\left|\hat{\mathbf{X}}_{j k}^{(t+i)}-\mathbf{X}_{j k}^{(t+i)}\right|$$

### 3. Experiments



## 2.写作

Spatial-temporal 时空

spatial relations/ temporal trend

seamlessly 无缝地

with the advance of ...

