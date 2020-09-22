# 论文笔记：Foundations and modelling of dynamic networks using Dynamic Graph Neural Networks: A survey

## 1.方法模型

### *Dynamic networks*

- definition

<img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200916151255466.png" alt="image-20200916151255466" style="zoom:80%;" />

-A.Dynamic network representations

<img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200916152146756.png" alt="image-20200916152146756" style="zoom:50%;" />

			- Static networks:have no temporal information
			- Edge weighted networks have temporal information included as labels on the edges and/or nodes of a static network. e.g.: edges labeled with the time they were last active
	
		- Discrete networks: multiple snapshots of the network at different time intervals
		- Continuous networks

1） Discrete Representation: Discrete representations use an **ordered set of graphs (snapshots)** to represent a dynamic graph.

$$D G=\left\{G^{1}, G^{2}, \ldots, G^{T}\right\}$$

allows for the use of static network analysis methods on each of the snapshots

other ways: Overlapping snapshots such as sliding time-windows, 不一定是 an ordered set of graphs, they may also be represented as a multi-layered network or as a tensor

2) Continuous Representation

We cover three continuous representations: (i) the event-based; (ii) the contact sequence; and (iii) the graph streams.

​	-event-based

$$E B=\left\{\left(u_{i}, v_{i}, t_{i}, \Delta_{i}\right) ; i=1,2, \ldots\right\}$$

​	-contact sequence 

link is instantaneous and thus no link duration is provided.

$$C S=\left\{\left(u_{i}, v_{i}, t_{i}\right) ; i=1,2, \ldots\right\}$$

​	-graph streams

it treats link appearance and link disappearance as separate events.

<img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200916154351887.png" alt="image-20200916154351887" style="zoom:67%;" />

-B.Link duration spectrum

different types. The scale goes from interactions with no link duration to links that have infinite link duration.

<img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200916155725710.png" alt="image-20200916155725710" style="zoom:70%;" />



<img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200916160233814.png" alt="image-20200916160233814" style="zoom:67%;" />

-C.Node dynamics

<img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200916160817394.png" alt="image-20200916160817394" style="zoom:67%;" />

-D.The dynamic network cube

-E.Dynamic network models

​	reference models & realistic model 

### *DYNAMIC GRAPH NEURAL NETWORKS*

dynamic graph neural network (DGNN), A key characteristic of a graph neural network is an aggregation of neighbouring node features (also known as message passing)

In the discrete case, a DGNN is a combination of a **GNN and a time series model**. Whereas in the continuous case we have more variety since the node aggregation can no longer be done using traditional GNNs

A.Pseudo-dynamic models

**G-GCN** can be seen as an extension of the Variational Graph Autoencoder (VGAE) [56] which is able to predict links for nodes with no prior connections, the so called cold start problem.



B.Discrete Dynamic Graph Neural Networks

two kinds of discrete DGNNs: Stacked DGNNs and Integrated DGNNs.

A discrete DGNN deep time-series modelling +GNN. The time-series model often comes in the form of an **RNN**, but **self-attention** has also been used.

<img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200916164308297.png" alt="image-20200916164308297" style="zoom:80%;" />

<img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200916164744327.png" alt="image-20200916164744327" style="zoom:67%;" />

**stacked DGNN**

GNN encodes each snapshots, RNN encodes across snapshots...

<img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200916165020226.png" alt="image-20200916165020226" style="zoom:67%;" />

<img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200916165228960.png" alt="image-20200916165228960" style="zoom:80%;" />

**Integrated DGNN**

<img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200916165542735.png" alt="image-20200916165542735" style="zoom:80%;" />

<img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200916165518242.png" alt="image-20200916165518242" style="zoom:80%;" />

**Dynamic graph autoencoders and generative models:**

动态图嵌入模型（DynGEM）使用深度自动编码器对离散节点动态动态图的快照进行编码。其主要思想是使用前一个快照的权重初始化自动编码器。这大大加快了计算速度，并使嵌入保持稳定（即从快照到快照没有重大变化）。为了处理新节点，在嵌入层保持不变的情况下，使用了[78]中的Net2WiderNet和Net2DeeperNet方法来*增加编码器和解码器的宽度和深度*。这使得自动编码器可以扩展，同时近似地保留神经网络正在计算的函数

Dyngraph2vec: considers the last l snapshots in the encoding and can thus be thought of as a sliding timewindow. The adjacency matrices At...At+l are used to predict At+l+1, it is assumed that no new nodes are added.



C.Continuous Dynamic Graph Neural Networks

- RNN based approaches

as soon as an event occurs or there is a change to the network, the embeddings of the interacting nodes are updated

**Streaming graph neural network**: maintains a hidden representation in each node, The architecture consists of two components: an update & a propagation component(更新组件负责更新参与交互的节点的状态，传播组件将更新传播到相关的节点邻居)

The update and propagation component consist of 3 units each: (i) the interact unit; (ii) the update / propagate unit; and (iii) the merge unit.

The model maintains several vectors for each node. (i) a hidden state for the source role of the node;
and (ii) a hidden state of the target role of the node.

**interact unit**: encoding of the interaction

**merge unit**: update the combined hidden state of nodes

**update unit**: generates a new hidden state for the interacting nodes[based on a Time-aware LSTM]

**propagate unit**: updates the hidden states of the neighboring nodes: consists of an attention function f, a time decay function g and a time based filter h. f estimates the importance between nodes, g gauges the magnitude of the update based on how long ago it was and h is a binary function which filters out updates when the receiving node has too old information.

**JODIE**:

JODIE uses an RNN architecture to maintain the embeddings of each node. With one RNN for users (RNNu) and one RNN for items (RNNi), the formula for each RNN is identical except that they use different weights. When an interaction happens between a user and an item, each of the embeddigs is updated according to equation

<img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200916173121339.png" alt="image-20200916173121339" style="zoom:80%;" />

- Temporal point process based models

DyREP:

 <img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200916173423344.png" alt="image-20200916173423344" style="zoom:80%;" />

### *Deep learning for prediction*

#### Decoders

time-conditioned decoder, A prediction is then often an adjacency matrix $A^{t1}$ which indicates the
probabilities of an edge at time t . Often t1 = t + 1. [预测t+1时刻的邻接矩阵]

H will need to be reshaped to produce $A^{t1}$， 一般用前馈神经网络来reshape，如果只考虑最后一个snapshot的，则不需要。这样就能得到Z，再从Z到$A^{t1}$， 可以用FFNN，也可以用inner product of the node embeddings：

$$p\left(A_{i j}^{t}=1 \mid z_{i}^{t}, z_{j}^{t}\right)=\sigma\left(\left(z_{i}^{t}\right)^{\top} z_{j}^{t}\right)$$

In general there are many options for how decoding can be done. As long as the **probability for each edge** is produced from the latent variables and the architecture can be efficiently optimized with back-propagation.

#### Loss functions

As the prediction methods optimize towards link prediction directly, an autoencoder optimizes towards recreation of the dynamic graph and also can be used and have been shown to perform well in link prediction tasks.

1)link prediction

Prediction of edges is seen as a **binary classification task**

traditional: extremely **unbalanced**

method: binary cross-entropy

some tricks: negative sampling, 这将链路预测问题从多输出分类（每个链路的预测）转换为二元分类问题（链路是“好”链路还是“坏”链路）。这加快了计算速度，并处理了链路预测中众所周知的类不平衡问题。

$$\mathcal{L}_{C E}=\sum_{i=1}^{n} \sum_{j=1}^{n} A_{i j}^{t} \log \left(\hat{A}_{i j}^{t}\right)$$

2)autoencoder 

旨在重建动态网络,所有被调查的自动编码器都在离散网络上工作。因此，将网络重构简化为对每个快照的重构。这需要创建一个损失函数来惩罚输入图的错误重建。变分自动编码器方法也旨在生成模型。为了生成，它们需要在潜在空间中启用插值。这是通过在损失函数中添加一个项来实现的，该项惩罚学习的潜在变量分布不同于正态分布。为了避免过度拟合，在损失函数中添加正则化

$$\mathcal{L}=\sum_{i=1}^{n} \sum_{j=1}^{n}\left(A_{i j}^{t}-\hat{A}_{i j}^{t}\right) * P_{i j}$$

<img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200916232927956.png" alt="image-20200916232927956" style="zoom:80%;" />

total loss function= reconstruction loss+regularization

$$\mathcal{L}_{\text {total }}=\mathcal{L}+\alpha \mathcal{L}_{\text {reg}}$$

变分自动编码器方法使用不同的正则化器。它们将节点嵌入规范化，。在传统的变分自动编码器中，这个先验是一个均值为0，标准偏差为1的正态分布。在动态图自动编码器中，先验仍然是高斯的，但它是由先前的观测值参数化的。

$$\begin{array}{l}
\mathrm{KL}\left(q\left(Z^{t} \mid A^{\leq t}, X \leq t, Z^{<t}\right)\right. \\
\left.\| p\left(Z^{t} \mid A^{<t}, X^{<t}, Z^{<t}\right)\right)
\end{array}$$

3)Temporal Point Processes

<img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200916233524340.png" alt="image-20200916233524340" style="zoom:80%;" />

#### Evaluation metrics

AUC,PRAUC,Fixed-threshold metrics.Mean Average Precision (MAP),Sum of absolute differences (SumD).Error rate;gmauc

## 2.写作

bulk 大量的

vast

interdisciplinary 跨学科的

dictate 命令，支配

trivially 平凡地，微不足道地

to the best of our knowledge

Fine grained 细粒度的

coarser 较粗的

persist 坚持

instantaneous  瞬间的

manifests 表明

replicate 复制模仿

be of relevance

magnitude 重要性

gauge 测量= measure

precursor

central

