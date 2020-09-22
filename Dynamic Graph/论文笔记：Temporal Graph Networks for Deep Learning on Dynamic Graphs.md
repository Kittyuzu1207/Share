## 论文笔记：Temporal Graph Networks for Deep Learning on Dynamic Graphs

### 1. 方法模型
（图的拓扑结构有改变的）

#### 1.Task & problem formulation

之前的一些limitations: setting of discrete-time dynamic graphs represented as a sequence of snapshots of the graph over time; Few approaches  support the inductive setting of generalizing to new nodes not seen during training

Dynamic Graph: 1) Discrete-time dynamic graphs (DTDG) are sequences of static graph snapshots taken at intervals in time. 2) Continuos-time dynamic graphs (CTDG) are more general and can be represented as timed lists of events, which may include edge addition or deletion, node addition or deletion and node or edge feature transformations.

This temporal graph is modeled as a sequence of time-stamped events: $G=\{{x(t_1),x(t_2)...}\}$, representing addition or change of a node or interaction between a pair of nodes at times

An event $x(t)$ can be **two_type** : 1) node-wise event $v_i(t)$  2)interaction event: a (directed) temporal edge $e_{ij}(t)$

A snapshot of the temporal graph $G$ at time t: $G(t)=(V[0,t],E[0,t])$ with $n(t)$ nodes.

#### 2. Models

a neural model for dynamic graphs can be regarded as an encoder-decoder pair: 

encoder: a function that maps from a dynamic graph to node embeddings

decoder: input one or more node embeddings and makes a prediction based on these, e.g. node classification or edge prediction 

in this paper, for each time t, the embedding of the graph nodes $Z(t)=(z_1(t),...,z_{n(t)}(t))$

#### Core modules

- **Memory**

  **node level**

  memory state of the model at time t consists of a vector $s_i(t)$ for each node i,  the memory node is updated when the node is envolved in an event 

  -->TGNs have the capability to memorize long term dependencies for each node in the graph

  **global memory**

  好处： 信息可以很容易地在图中传播很远的距离，节点的memory可以随着全局状态的变化而更新，基于全局memory的简单的graph-wise预测, 但在本项工作中不涉及

- **Message Function**

  用一个function去update node的memory

  **interaction** event $e_{ij}(t)$ between node i and node j at time t, two messages can be
  computed for the source and target nodes that respectively start and receive the interaction:

  $$\mathbf{m}_{i}(t)=\operatorname{msg}_{\mathrm{s}}\left(\mathbf{s}_{i}\left(t^{-}\right), \mathbf{s}_{j}\left(t^{-}\right), t, \mathbf{e}_{i j}(t)\right), \quad \mathbf{m}_{j}(t)=\operatorname{msg}_{\mathrm{d}}\left(\mathbf{s}_{j}\left(t^{-}\right), \mathbf{s}_{i}\left(t^{-}\right), t, \mathbf{e}_{i j}(t)\right)$$

  **node-wise** event : $v_i(t)$

  $$\mathbf{m}_{i}(t)=\operatorname{msg}_{\mathrm{n}}\left(\mathbf{s}_{i}\left(t^{-}\right), t, \mathbf{v}_{i}(t)\right)$$

  $t^-$ 是指just before time t, msg 是learnable message function

  msg: e.g. MLPs

- **Message Aggregator**

  multiple events to a same node in a same batch,  use a mechanism to aggregate messages

  $$\overline{\mathbf{m}}_{i}(t)=\operatorname{agg}\left(\mathbf{m}_{i}\left(t_{1}\right), \ldots, \mathbf{m}_{i}\left(t_{b}\right)\right)$$

  agg is an aggregation function, for the sake of simplicity we considered
  two efficient non-learnable solutions in our experiments: most **recent** message (keep only most recent message for a given node) and **mean** message (average all messages for a given node). 其他方法在以后讨论

  e.g. RNNs, attention w.r.t

- **Memory Updator**

  the memory of a node is updated upon each event involving the node itself：

  $$\mathbf{s}_{i}(t)=\operatorname{mem}\left(\overline{\mathbf{m}}_{i}(t), \mathbf{s}_{i}\left(t^{-}\right)\right)$$

  mem: learnable function, e.g. LSTM, GRU

- **Embedding**

used to generate the temporal embeddings $z_i(t)$ of node i at time t, 

the main goal of the embedding module: avoid the so-called staleness problem.( 数据过时？),  it might happen that, in the absence of events for a long time (e.g. a social network user who stops using the platform for some time before becoming active again), i’s memory becomes stale

use:

$$\mathbf{z}_{i}(t)=\operatorname{emb}(i, t)=\sum_{j \in n_{i}^{k}([0, t])} h\left(\mathbf{s}_{i}(t), \mathbf{s}_{j}(t), \mathbf{e}_{i j}, \mathbf{v}_{i}(t), \mathbf{v}_{j}(t)\right)$$

h: learnable function, This includes many different formulations as particular cases:

1) identify(id): $emb(i,t)=s_i(t)$ 直接用memory

2)Time projection (time): $emb(i,t)=(1+\Delta tw)\circ s_i(t)$, w is learnable parameters,  $\Delta t$ is the time since the last interaction.  $\circ$ is element-wise vector product.

3)Temporal Graph Attention (attn): by aggregating information from its L-hop temporal neighborhood 

<img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200914170615887.png" alt="image-20200914170615887" style="zoom:80%;" />

$\phi()$  is a generic time encoding, || is concatenation operator, and **$z_i(t)=h_i^{(L)}(t)$**

each layer amounts to a multi-head attention,  query: a reference node, keys&values: its neighbors

input representation of each node: $h_j^{(0)}(t)=s_j(t)+v_j(t)$

4) Temporal Graph Sum (sum): A simpler and faster aggregation over the graph:

$$\begin{aligned}
\mathbf{h}_{i}^{(l)}(t) &=\operatorname{MLP}^{(l)}\left(\mathbf{h}_{i}^{(l-1)}(t) \| \tilde{\mathbf{h}}_{i}^{(l)}(t)\right), \\
\tilde{\mathbf{h}}_{i}^{(l)}(t) &=\sum_{j \in n_{i}([0, t])} \mathbf{h}_{j}^{(l-1)}(t)\left\|\mathbf{e}_{i j}\right\| \phi\left(t-t_{j}\right)
\end{aligned}$$

here, $$\mathbf{z}_{i}(t)=\operatorname{emb}(i, t)=\mathbf{h}_{i}^{(L)}(t)$$

#### Training

We present two possible training procedures for TGNs while using the  **link prediction** task as a simple example: provided a list of **ordered timed interactions**, the goal of the model is to **predict the future interactions** from those observed in the past.

<img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200914123654152.png" alt="image-20200914123654152" style="zoom:80%;" />

--interactions serve two purposes: 1) training objective 2) 在预测一个batch的某条边之前，这些interactions 不能用于更新memory，否则会导致信息泄露，但是如果调换操作顺序（在更新memory之前先计算loss和预测）则会导致所有与memory相关的模块都无法接收梯度，因此需要一些其他的步骤：

<img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200914173117594.png" alt="image-20200914173117594" style="zoom:80%;" />

- Basic training strategy:

order: predict interactions, then update memory, but breaks every batch2 of size b into k sub-batches of size b=k. The sub-batches are processed sequentially with their losses accumulated and backpropagation is only performed after the last sub-batch. 

<img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200914173343694.png" alt="image-20200914173343694" style="zoom:80%;" />

two drawbacks:

it slows down the training, as each batch is not computed fully in parallel; the only nodes that contribute to the memory-related modules’ gradients are those with at least one interaction in multiple sub-batches

- Advanced training strategy:

reverse the order of operations , $\tilde{t}_{i_{i}}$ is the time of node i’s last interaction in its last sub-batch  $b_i(\tilde{t}_{i_{i}})$

store memory $\mathbf{s}_{i}\left(\tilde{t}_{i}^{-}\right)$, the state of i prior to the last sub-batch, together with the raw information

<img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200914175610895.png" alt="image-20200914175610895" style="zoom:80%;" />

#### 3.Experiments

##### Settings

- Dataset: Wikipedia, Reddit , and Twitter, The features of an interaction are a BERT-based [63] vector
  representation of the text of the retweet.
- Tasks: future edge prediction and dynamic node classification. transductive settings: predict future links of nodes which were observed during training;  inductive settings: predict future links of nodes never observed before
- Future Edge Prediction: predict the probability of an edge occurring between two nodes at a given time, Our encoder is combined with a simple MLP decoder mapping from the concatenation of two node embeddings to the probability of the edge.
- Dynamic Node Classification: The task is to predict a binary label indicating whether a user was
  banned at a specific time. We pre-train our encoder on the future edge prediction task, then freeze it
  and combine it with a task-specific MLP decoder

##### Choice of modules

- Memory:  TGN-no-mem, TGN-attn
- Embedding: TGN-id, TGNtime, TGN-attn, TGN-sum



### 2.写作

ranging from... to...

can be cast as specific instances of

ubiquitously 无处不在地

message passing mechanism

sub-optimal

generic 通用的

the sequentiality of the data 数据的连续性

be endowed with 被赋予

envisage the benefits 设想...带来的好处

for the sake of 为了

staleness 