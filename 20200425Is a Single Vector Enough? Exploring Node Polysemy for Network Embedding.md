# Is a Single Vector Enough? Exploring Node Polysemy for Network Embedding 一个向量就够了吗？网络嵌入中的节点多义性探讨
## Abstract  
网络嵌入模型是将网络中的节点映射成连续向量空间表示的有力工具，以便于后续的分类和链路预测等任务。现有的网络嵌入模型将每个节点的所有信息，如链接和属性，综合集成到一个嵌入向量中，以表示节点在网络中的一般角色。然而，一个现实世界的实体可能是多方面的，它由于不同的动机或自我特征而连接到不同的社区，而这些动机或自我特征并不一定相关。例如，在电影推荐系统中，用户可以同时喜爱喜剧片或恐怖片，但这两种类型的电影在嵌入空间中不太可能相互接近，用户嵌入向量也不可能同时足够接近它们。本文针对语言建模中的多义现象，提出了一种基于多义词嵌入的节点多方面建模方法。节点的每一个方面都映射到一个嵌入向量，同时在每一对节点和方面之间保持一个关联度。该方法对现有的各种嵌入模型具有自适应性，且不会使优化过程复杂化。我们还讨论了如何利用不同方面的嵌入向量进行分类和链接预测等推理任务。在真实数据集上的实验有助于综合评价该方法的性能。  

## 1 INTRO
网络是用于信息系统建模的普遍存在的数据结构，如社会网络、推荐系统、生物网络和知识图。在这些系统中，用户、项目、分子和知识概念等现实世界实体被抽象为网络中的节点，而实体之间的关系则被建模为它们之间的链接。网络嵌入的最新进展建议，通过考虑节点的neighborhood和特征信息feature info，将每个节点表示为低维向量。在嵌入空间中，相似的节点被紧密地映射在一起。节点嵌入已经被证明是一种有效的表示方案，有助于下游网络分析任务，如分类、聚类和链路预测。  
在许多实际应用中，实体可能具有不同的特征或方面。换句话说，网络中的节点可以被视为包含不同方面的capsule。事实上，从一个实体延伸到它们的邻居的不同链接可能是由其不同方面的表现造成的。在某些场景中，将这些不同的方面融合到节点的单个向量空间表示中可能会有问题。例如，在图1中客户和项目是节点的在线购物网站中，客户可能购买了不同类型的项目iitems。如果我们只用一个向量来表示每个客户，那么客户和项目的嵌入向量必须同时彼此接近。这可能很难实现，因为其他客户有不同的兴趣，这可能会扰乱嵌入向量的分布。  
![img](https://github.com/Kittyuzu1207/Share/blob/master/img/0426pic1.png)  
通过类比自然语言中单词所具有的相似性质（例如，“bank”可以指金融机构或河流附近的土地，具体取决于不同的上下文）本文将这种现象称为节点多义现象，即每个节点可以有多个方面。在此设置中，每个节点都有多个方面，而节点的每个方面都拥有嵌入向量。在这项工作中，我们希望开发一个多义网络嵌入方法，以发现节点的多个方面，并学习它们的表示。  
### Challenges
开发多义网络嵌入模型的挑战有三个方面。
- 首先是如何确定节点的facet，以及灵活地更新向量空间中不同facet的嵌入。对于每个数据样本（例如，链接links或随机游走random walk），我们需要确定每个节点的哪个方面可能被激活，以便在训练过程中更新该方面的相应嵌入。
- 第二个挑战是如何保持不同方面的嵌入向量之间的相关性。尽管我们将一个节点分成多个面，但不同的面可能并不完全相互不相关。如果我们简单地对每个方面分别建模，一些信息将丢失。
- 第三，在考虑节点多义性的情况下，如何使建模过程适应Deepwalk、LINE、PTE和GCN等已有的基础模型。此外，当考虑节点的不同侧面时，新模型的计算复杂度必然增加。因此，还需要一种有效的优化算法，特别是考虑到它与负采样的兼容性。  
具体来说，本文提出了一种多义网络嵌入方法，以考虑网络数据中节点的多个方面。节点的每个方面都使用嵌入向量表示。我们的方法可以保留不同节点面之间的相关性。所开发的建模策略灵活地应用于各种基础嵌入模型，而不必对其基础公式进行根本更改。我们首先展示如何修改Deepwalk来解决节点多义问题，然后将我们的讨论扩展到更多的基础嵌入模型和更复杂的应用场景。为了保证算法的有效性和实现的可行性，特别是在采用负采样的情况下，对多义嵌入模型的训练过程进行了优化设计。最后，为了评估考虑多个方面的节点是否有利于学习节点表示和下游数据挖掘任务，我们对不同任务进行了实验，比较了多义嵌入模型和相应的单向量基模型的性能。  
### Contributions
- 提出了一种新的多义网络嵌入方法，将节点的不同方面融入到表示学习中。节点的每个方面都有一个嵌入向量，并考虑了不同方面的嵌入向量之间的关系。  
- 我们将问题具体化，以使所得到的优化算法能够有效且可行地实现。该方法对现有的各种反导(transductive)嵌入模型具有很好的适应性。
- 我们针对不同的下游任务和应用场景进行了深入的实验，提供了如何以及何时从网络嵌入中的节点多义性建模中获益的见解。

## 2 POLYSEMOUS NETWORK EMBEDDING
本部分以Deepwalk为基础模型，介绍了多义网络嵌入的核心思想。然后，我们设计了一个优化算法来训练多义词嵌入模型。我们还将讨论如何在不同的上下文中估计节点的方面。最后，我们介绍如何将不同方面的嵌入向量组合到下游任务中，例如分类和链接预测。
### 2.1 Polysemous Deepwalk
在多义嵌入polysemous embedding的设定中，每个节点v_i和一个target embedding matrix U_i (K_i×D) 和一个context embedding matrix H_i(K_i×D)相联系。K_i是节点v_i拥有的embedding vectors的数量，考虑到其不同的facets。传统的DeepWalk model中K_i=1。D是embedding的维度，节点v_i的第k个facet的embedding vector 表示为 {U_i}^k 或{H_i}^k。不同的节点可以与不同数量的嵌入向量相关联，这取决于它们在网络中的特性的多样性。在这项工作中，为了举例说明，我们简单地让所有节点具有相同数量的嵌入向量，这样K i=K和K是一个预定义的常量整数。实际上，K的值可以由数据来估计。例如，K可以近似地设置为推荐系统中潜在兴趣类别的数目，也可以估计为学术网络中主要主题的数目。我们将在后面的部分讨论如何将facet分配给具有不同概率的节点。  
***
DeepWalk 使用了Skip-gram 模型,该模型使用最大似然估计进行训练，在此我们试图找到使获得的观测值的似然最大化的模型参数。具体来说，设θ为待优化参数，O为所有观测值的集合，待最大化的目标函数为：  
![img](https://github.com/Kittyuzu1207/Share/blob/master/img/04262.png)   
其中每个observation o属于O，是一个tuple，o=(N(v_i),v_i)，由一个central node v_i 和其context组成，在这个context里的node表示为v_j,所以v_j 属于 N(v_i).model parameters e.g. the embedding vectors of nodes 被用来计算概率p(v_j|v_i),这个条件概率是给定v_i的条件下v_j出现在其context里的概率  
但是，在我们的设置中，每个节点拥有多个facets，activated facets of a node在不同的上下文中有所不同。此外，给定上下文的facet由上下文中节点的所有facets的组合确定。假设节点facets的分布是预先知道的，我们把它们当作表示为P的先验知识。考虑到其他信息，目标重新表述为：  
![img](https://github.com/Kittyuzu1207/Share/blob/master/img/04263.png)  
其中s(o)表示o中所有节点activated facets的一个case，所以s(o)={s(v|o) |v ∈ v i ∪ N(v i )} 其中s（v | o）是o上下文中节点v的activated facets。在给定的观测值o中，假设v_i的activated facet是k_i，且每个v_j∈N(v_i)的activated facet是k_j，则条件概率p（o|s(o),p,θ）定义为：  
![img](https://github.com/Kittyuzu1207/Share/blob/master/img/04265.png)   
其中每一个product vector按上图计算，类似于传统的skip-gram模型中的softmax归一化，只是原本的node embedding变成了 node facet embeddings。这里的<,>表示两个向量的内积。分母作为一种归一化over all possible nodes and facets。为了可读性，省略上面公式中的P，之后不会用了。  
***
由于对数函数中存在求和项，直接应用梯度下降法优化方程2∼方程4中的目标函数比较麻烦。此外，如何结合负采样[22]来近似归一化项以提高计算效率也变得不清楚。因此，我们进一步推导出目标函数如下  
![img](https://github.com/Kittyuzu1207/Share/blob/master/img/04266.png)   
这种转换背后的直觉是， instead of maximizing the original objective function, we turn to maximize its lower bound 使用Jesen函数使其下界最大化。表示为上面这个亚子。除了s(o)的外求和external summation over s(o)之外，目标函数与skip gram model的类似，因此也可以采用与传统skip gram模型相同的负采样策略来近似p（vj|vi,s(o)）中的归一化项。因此，所提出的多义点嵌入模型的一个主要优点是，通过对现有学习框架的最小修改，通过增加一个额外的采样步骤 of assigning activated facets to nodes in each observation o，可以很容易地实现训练过程。  
具体而言，给定p(s(o)|P)的分布，对(s(o))的求和项是通过facet sampling separatelyfor each node in o 来实现的。算法1中给了总体优化算法。  
![img](https://github.com/Kittyuzu1207/Share/blob/master/img/04267.png)   
在初始化和节点面分布估计node-facet distribution estimation（稍后将介绍）之后，像在传统的Deepwalk（第3行）中一样，对一些随机游动进行采样。然后，对于每个观测o，在o（第7行中的循环）内的每个节点上进行多轮刻面采样several rounds of facet sampling are conducted on each node within o。在每一轮中，每个node都激活了一个facet（第8∼10行），这样对应于该facet的嵌入向量将使用SGD进行更新（第11行）。与传统的Deepwalk相比，主要的额外计算成本来自于O中的训练数据增加了采样率R的一个因子**R**。  
多义Deepwalk的整个过程如图2所示，其中如上所述引入了“目标优化”，而有关方面分布和方面分配的其他步骤将在下一小节中详细讨论。  
![img](https://github.com/Kittyuzu1207/Share/blob/master/img/04264.png)   

### 2.2 Node-Facet Assignment 结点-方面分配
现在我们将介绍如何获得先验知识P，以及如何在给定特定观测o的情况下确定节点的facet。目前，我们将讨论限制在无向齐次平面网络undirected homogeneous plain networks，并提供了一种仅利用网络邻接矩阵来获得节点面全局分布的方法，对于具有属性信息的网络或异构信息网络，我们可以采用其他策略，并将其留给以后的工作。设A为网络的对称邻接矩阵，在网络上进行社区发现[15][36]：
![img](https://github.com/Kittyuzu1207/Share/blob/master/img/04268.png)   
其中 **P** 可以用梯度搜索算法，概率p(k|v)即结点v分配了第k个facet的概率，可以通过公式xxx计算。计算一个node的 facet distribution，as p(v) = [ p( 1 |v),...,p(K|v) ]。我们把p(v_i)，1≤i≤N作为先验知识p，因为它编码了对网络状态的全局理解。应用属性信息可以得到更好的估计，但超过了本文的讨论范围。   
***
在给定先验知识P后，我们能够估计出overall facet of each observation o as well as the facet of nodes within o。给定观测值o=（N（vi），vi），一个直接的方式是通过对the facet distributions of nodes inside the observation取平均，来获得its facet distribution p(o) 。即公式xxx。考虑到节点的activated facet依赖于观察到它的特定上下文，给定o，节点的facet s(v|o)根据分布p(v|o)进行采样，该分布p（v|o）的启发式定义如下:公式xxx。其中，我们引入min（·，·）运算符，因为如果p（k|vi）≈0，则不希望使用facet k分配节点vi，即使p(o)中的第k个条目很大。为了使其成为有效的概率分布，p（v|o）进一步规范化为和为1。   
到目前为止，我们已经介绍了多义词Deepwalk的整个训练过程，如图2所示。在给定输入网络的情况下，我们首先将节点刻面分布估计为先验知识P。然后，执行随机游动来构造节点上下文观测。之后，在每个walk示例o中，为每个节点分配一个激活的facet。最后，通过优化更新相应面的节点嵌入。  

### 2.3 Joint Engagement of Multiple Embeddings for Inference
在训练多义模型后，我们可以得到每个节点的多个嵌入向量。接下来的问题是，在推理过程中，如何为后续任务共同考虑不同的嵌入向量。这里我们讨论两个主要的网络分析任务，包括分类和链路预测。  
对于分类任务，对于每个node，我们的策略是combine multiple vectors into a joint vector。有一些options 比如：直接concat，或者first scale each embedding vector with the probability of belonging to the corresponding facet再concat。连接后的resultant embeddin可以直接用于node classification。我们采用的是先scale再concat。   
对于链路预测或网络重建任务，两个节点表示之间的相似度得分越高，表明节点之间存在链路的可能性越大，可以将两个节点之间的相似度定义为：公式10。其中嵌入向量的不同facet对有助于整体相似性计算，由节点属于相应方面的概率加权。  

### 2.4 Discussion
由于不同的表征维度对数据背后的不同因素敏感，因此本文的工作可能与分离表征学习相关[9]。此外，它有助于提高表示学习的可解释性[3]，因为表示维度是根据可能与实际网络对象的具体含义或特征相关联的节点方面来分离的。  

## 3 MODELS EXTENDED BY POLYSEMOUS EMBEDDING







