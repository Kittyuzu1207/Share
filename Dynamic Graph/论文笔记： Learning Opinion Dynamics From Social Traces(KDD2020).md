## 论文笔记： Learning Opinion Dynamics From Social Traces(KDD2020)

### 1.方法模型

#### 1. Task &problem formulation

Opinion dynamics –the research field dealing with how people’s opinions form and evolve in a social context

传统方法：使用agent-based models 来验证社会学理论的含义

这项工作提出了一个推理机制，以fit一个生成的，agent-like的意见动态模型，以适应现实世界的社会痕迹，在给定一组可观察值（例如，agent之间的行为和交互作用），我们的模型可以恢复最有可能的潜在意见轨迹，这些轨迹与关于过程动力学的假设相一致。这种模型保留了agent-based的模型（即因果解释）的优点，同时增加了对实际数据进行模型选择和假设检验的能力。我们通过将一个经典的基于agent的意见动态模型转换为它的生成模型，然后设计了一种基于在线期望最大化的推理算法来学习模型的潜在参数。该算法可以从经典的基于agent模型生成的轨迹中恢复潜在的意见轨迹。此外，它还可以识别用于生成数据跟踪的最有可能的一组宏参数，从而允许对社会学假设进行测试。

- ABMs: agent-based models, 允许从经验上探索社会学假设的含义，并将其形式化为agent之间的互动规则, 是mechanistic model，很容易用因果关系解释（不同于统计的方法），但是预测能力有限&参数校准难&难以理解个人级别的digital trace

- 提出的模型: Learnable Opinion Dynamics model，LODM, 它保留了ABMs的因果推断机制，同时允许从实际数据中进行参数推断

#### 2. Model

##### GENERATIVE FRAMEWORK

interactions between agents are the driver of opinion change

are not easily observable: the opinion of a single agent /the “sign" of the interactions 

observable: interaction between two agents has happened / actions performed by individuals :hashtag, Reddit community

- Observables[可观测量]

$V$ : a set of actors, who interact and influence each other’s opinion

interactions: a temporal graph, $T$ discrete time steps, each actor is a node

[different opinions lead to different actions (think, for instance, of putting a “like” on a politically-charged Facebook page).]

$A$ : be the set of possible actions

--> construct a bipartite graph

$G=(V,E)$ : directed interaction graph between actors, arc $(u,v,t)\in E$ : user u interact with user v at time t

$Z=(V,A,F)$ : bipartite graph of actors and actions, arc $(v,a,t) \in F$ : represents that “actor v performs action a at time t ”.

一个example <img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200913173238609.png" alt="image-20200913173238609" style="zoom:67%;" />

- Latent variables

each interation in G is either positive or negative

action 改变actor 的latent opinions

<img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200913175629404.png" alt="image-20200913175629404" style="zoom:80%;" />



- Base  Model（deterministic ABM)

determined by two macro parameters:  a latitude of acceptance and a latitude of contrast： $\epsilon^{+}$ 和$\epsilon^{-}$ 

The sign of an interaction (u,v, t ) ∈ E is determined when u expresses its opinion to v: 

if close(within $\epsilon^{+}$), v accept it; if distant (further than $\epsilon^{-}$), v constracts it

<img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200913180710625.png" alt="image-20200913180710625" style="zoom:80%;" />

The parameters μ+, μ− > 0 thus control the speed of the influence due to the interactions.

- Generative process for **interactions**

deterministic--> its probabilistic generative counterpart

in this model: (1) a node u at time t generates a given, fixed number γt,u of arcs  (2)at
each time step t , only a subset $V _t^* \in V$ of the nodes is considered active and eligible to receive an arc.

为了使interaction具有随机性，我们首先需要确定一个interaction在时间t为positive的先验概率

![image-20200913181931043](C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200913181931043.png)

define the probability of an interaction (u,v, t ) as a function of the opinions of u and v, and of the sign of the arc.

<img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200913182357464.png" alt="image-20200913182357464" style="zoom:80%;" />



<img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200913182532304.png" alt="image-20200913182532304" style="zoom:80%;" />



- Generative process for **actions**

  <img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200913182926207.png" alt="image-20200913182926207" style="zoom:80%;" />

  <img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200913183023626.png" alt="image-20200913183023626" style="zoom:80%;" />



##### Learning

提出了一个最大化模型完全数据似然的算法，在给定观测值和宏观参数的情况下估计潜在变量。

- Complete-data likelihood

把数据集的完全似然$P(E,F)$ 写成$P(E)\cdot P(F)$

interaction likelihood :P(E)   action likelihood: P(F)

P（E）可分解为正、负相互作用两种互斥的情形:

<img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200913232656033.png" alt="image-20200913232656033" style="zoom:80%;" />

<img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200913232807976.png" alt="image-20200913232807976" style="zoom:80%;" />



<img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200913233305926.png" alt="image-20200913233305926" style="zoom:80%;" />



优化该函数并不简单，因为潜在变量表达式Θ包含S（交互图中每个弧的符号）是一个离散变量，因此会导致整数规划问题。我们不能通过符号的标准线性松弛来解决这个问题，因为这意味着定义接受和对比之间的情况。因此使用EM

- Online EM

choose a set of parameters $\theta=(x_0, w, \sigma)$ from latent variables $\Theta$, wish to maximize the joint distribution $P(E,F|x_0,w,\sigma)$ given observed variables $\Omega=(E,F)$, latent variable S and parameters $\theta$

online task: at each time step, our algorithm is presented with new interactions $E_t$

-->needs to decide their sign.(pos& neg)-->update its estimate for the opinions of the actors

![image-20200914095025386](C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200914095025386.png)

(online assumptions)

define following expectation-maximizatoon steps:

![image-20200914095305677](C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200914095305677.png)

<img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200914100557854.png" alt="image-20200914100557854" style="zoom:80%;" />



#### 3. Experiments

-- Recovering opinion micro parameters

$\epsilon^{+}$和$\epsilon^{-}$的确定

--Discriminating macro-level scenarios

whether our framework is able to discriminate which scenario generated a given data trace，给定真实数据，看每个测试场景下生成的每个不同数据跟踪的可能性

--Opinion dynamics on real data

explore the prominence of the backfire effect, i.e., to see whether a scenario with large latitude of contrast is likely.

![image-20200914104549323](C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200914104549323.png)

### 2.写作

### 词

data traces

simulation



### 句

has recently received growing attention

access and consume an immense amount of content

