# Multi-Modal Sarcasm Detection in Twitter with Hierarchical Fusion 多模态图文反讽识别

## Abstract 
以往与讽刺相关的研究集中于文本。但是现在越来越多的社交平台如Twitter能够让用户创作多模态的信息，包括图片视频等等。所以仅仅依靠文本分析是不够的。  
在这篇文章中，作者吧文本特征、图片特征和图片属性作为三个模态，并且提出了一个多模态层次融合的方法来解决这个问题。模型首先提取图像特征和属性特征，然后利用属性特征和双向LSTM网络提取文本特征。然后重建三种模式的特征，并将其融合到一个特征向量中进行预测。此外，建立了一个基于Twitter的多模态讽刺检测数据集。数据集上的评估结果证明了提出的模型的有效性和三种模态的有用性。

## 1 Intro
讽刺的定义：一种讽刺性措辞，取决于它对刻薄、刻薄和经常讽刺的语言的影响，这种语言通常是针对个人的。能够掩饰说话者的敌意。  
讽刺在当今的社交媒体平台上普遍存在，其自动检测在客户服务、意见挖掘、网络骚扰检测以及各种需要了解人们真实情感的任务中具有重要意义。  
之前基于Twitter的讽刺检测工作主要集中在文本特征上，提出了许多有监督的方法，包括具有词汇特征的传统机器学习方法(Bouazizi and Ohtsuki, 2015;
Ptáˇ cek et al., 2014)以及深度学习的方法  
然而，只基于文本特征会存在一些问题。比如，像“What a wonderful weather!” 配图一个很阴沉天气这样的讽刺就无法识别出来，而图片能够帮助我们确定tweet文本是否有反讽。如下面a),b)  
![img](https://github.com/Kittyuzu1207/Share/blob/master/img/04171.png)  
在这篇文章中，作者提出了多模态层次融合模型来检测反讽。选择了三个方面的特征： text, image and image attribute features  
- 早期的融合： the attribute features are used to initialize a bi-directional LSTM network (Bi-LSTM)，再用Bi-LSTM来捕捉文本特征  
- 然后对这三个特征进行表示融合，并将其转化为重构的表示向量。  
模态融合层对向量进行加权平均，并将其给到分类层以得到最终结果。结果表明，这三种特征都有助于提高模型的性能。此外，融合策略成功地细化了每种模式的表示，并且比简单地将三种类型的特征连接起来更有效。
主要的贡献：  
- 提出了一个新颖的层次融合模型来解决Twitter中具有挑战性的多模态讽刺检测任务。据我们所知，我们是第一个深入融合图像、属性和文本这三种模式的人，而不是简单的串联，用于Twitter讽刺检测。  
- 建立了一个多模态推特反讽检测的数据集。
- 定量地展示了每种方式在Twitter讽刺检测中的意义。

## 2 Related Work
### 2.1 Sarcasm Detection
从文本中检测讽刺的方法已有很多。  
早期的方法从文本中提取一些手工特征(Davidov et al., 2010; Riloff et al., 2013; Ptáˇ cek
et al., 2014; Bouazizi and Ohtsuki, 2015) 包括：n-grams, word’s sentiment, punctuations, emoticons, part-of-speech tags
（n-gram，情感词，标点，表情符号，词性）  
最近：使用深度学习的方法来获取文本特征。Ghosh and Veale (2016):CNN+RNN layers  
除了tweet文本本身，也有人关注到一些语境特征contextual features，比如发布者的历史行为：
  - Bamman and Smith (2015)：发布者和回复者的特征+回复的文本特征  
  - Zhang，Zhang和Fu（2016）：讲目标tweet的embedding和特征concat起来  
  - Poria et al. (2016)：在之前的基础上，串联情感、情绪、个性
  - Y. Tay et l. (2018)：提出了多维的内注意力机制(intra-attention)，以明确地对比不一致
  - Wu et al. (2018)：基于embedding、情感特征和句法特征，构建一个具有密集连接LSTM的多任务模型。
  - Baziotis et al. (2018)：集成基于单词的双向LSTM和基于字符的双向LSTM，以捕获语义和句法特征。
然而，到目前为止，关于如何有效地结合文本和视觉信息来提高Twitter讽刺检测性能的研究还很少。Schifanella et al（2016）简单地将文本和图像的人工设计特征或基于深度学习的特征串联起来，用两种模式进行预测。与此不同的是，我们提出了一个层次化的融合模型来深度融合三种模式。

### 2.2 Other Multi-Modal Tasks
情感分析是一项与讽刺检测相关的任务。许多关于多模态情感分析的研究涉及视频数据（Wang et al,2016; Zadeh et al,2017），其中文本、图像和音频数据通常可以相互对齐和支持。虽然输入是不同的，但它们的融合机制可以激励我们完成任务。Poria、Cambria和Gelbukh（2015）使用多核学习来融合不同的模式。Zadeh et al（2017）通过外部产品构建它们的融合层，而不是简单的连接，以获得更多的功能。Gu et al（2018b）在单词级对齐文本和音频，并应用多种注意机制。Gu et al(2018a）首先介绍模态融合结构，试图揭示多模态的实际重要性，但它们的方法与我们的分层融合技术有很大不同。

还可以从其他多模态任务中获得灵感，例如视觉问答（VQA）任务，其中提供图像框架和查询语句作为模型输入。VQA任务（Chen et al,2015）中提出了一种问题引导注意机制，与使用全局图像特征的任务相比，该机制可以提高模型的性能。引入属性预测层（Wu et al,2016年），将高级概念纳入CNN-LSTM框架。Wang et al（2017）利用一些现成的算法，将它们与共同关注模型结合起来，实现可概括性和可伸缩性。Yang et al（2014）尝试使用带有图像注释的图像情感提取任务，并通过学习Bernoulli参数，提出了一种连接图像和注释信息的模型。

## 3 Proposed Hierarchical Fusion Model
图2显示了我们提出的分层融合模型的架构。在这项工作中，我们将文本、图像和图像属性视为三种模态。图像属性模态通过添加图像内容的高级概念来提高模型性能（Wu et al,2016）。为了充分利用这三种模态，提出了模态融合技术。在下文中，将首先定义原始向量raw vectors和制导向量guidance vectors，然后简要介绍我们的分层融合技术。  
![img](https://github.com/Kittyuzu1207/Share/blob/master/img/04172.png)  
- 对于图像模态，我们使用一个经过预训练和微调的ResNet模型来获得tweet图像的14×14区域向量，这些区域向量被定义为原始图像向量，并对它们进行平均以得到我们的图像引导向量(image guidance vector)  
- 对于（图像）属性模态，我们使用另一个预先训练和微调的ResNet模型来预测每个图像的5个属性，其中Glove embeddings被视为原始属性向量。我们的属性引导向量attribute guidance vector是原始属性向量的加权平均值。
- 使用Bi-LSTM来获得文本向量。原始文本向量是Bi-LSTM的每个时间步的前向和后向隐藏状态concat起来的，而文本引导向量 text guidance vector是上述原始向量的平均值。
基于附加图像有助于模型对tweet文本的理解，我们对属性引导向量进行非线性变换，并将结果作为Bi-LSTM的初始隐藏状态。这个过程叫做早期融合。（early fusion）。为了利用多模态信息来细化各模态的表示，提出了利用原始向量和引导向量重构三种模态特征向量的表示融合方法。在模态融合过程中，将三种模态的精向量合并为一个加权平均向量，而不是简单的级联。最后，将融合后的向量抽入两层全连通神经网络，得到分类结果。模型的细节如下：

### 3.1 Image Feature Representation 图象特征表示
使用ResNet-50 V2 (He et al., 2016)来获得tweet图象的表征。为了对模型进行微调，将预训练模型的最后一个完全连接层用一个新的FC层替换。参考(Wang et al., 2017)的工作，将输入图像 $I$ 重新调整为448×448，并划分为14×14的区域。每个区域 $I_i$ （i=1,2…，196）然后通过ResNet模型发送以获得区域特征表示$v_{region_i}$ ，即原始图像向量。  
$$v_{region_i}=ResNet({I_i))$$  
制导向量image guidance vector是各个region原始向量的平均：
$$v_image=\frac{{\displaystyle \sum^{i=1 \to \{N_r}}{v_{region_i}}{N_r}$$  
$N_r$是regions的数量，论文中是196。

### 3.2 Attribute Feature Representation 属性特征表示
之前在图像字幕和视觉问答中的工作（Wu et al，2016年）引入图象属性作为图像的高级概念high-level concepts。在他们的工作中，提出了单标签和多标签损失来训练属性预测CNN，其参数被用来生成最终的图像表示。虽然它们使用参数共享来完成带有属性标记任务的图像表示，但我们采用了更明确的方法。我们将属性作为连接tweet文本和图像的额外形式，直接使用每个tweet图像的五个预测属性的单词嵌入作为原始属性向量。  
我们首先使用ResNet-101和COCO图像字幕数据集训练属性预测器（Lin et al,2014）。我们通过从COCO数据集的句子中提取1000个属性来构建多标签数据集。我们使用在ImageNet上预先训练的ResNet模型（Russakovsky et al,2015），并在多标签数据集上对其进行微调。然后使用属性预测器预测每个图像的五个属性ai（i=1，…，5）。  
利用加权平均法生成属性引导向量。原始属性向量$e(a_i)$通过两层神经网络获得用于构造属性引导向量$v_attr$的注意权重$α_i$。相关方程如下:  
![img](https://github.com/Kittyuzu1207/Share/blob/master/img/04173.png)  
其中$α_i$是第i个图象属性，字面意义上就是从1000个单词中选出一个，e 是 GloVe embedding operation,$W_1$和$W_2$是权重矩阵,$b_1$和$b_2$是偏差bias，$N_a$是属性的数量，在这里取5  

### 3.3 Text Feature Representation 文本特征表示
使用双向LSTM（bilstm）（Hochreiter and Schmidhuber,1997）获取tweet文本的表示。LSTM在时间步t执行的操作方程式如下：  
![img](https://github.com/Kittyuzu1207/Share/blob/master/img/04174.png)  
其中$W_i$,$W_f$,$W_o$,$U_i$,$U_f$,$U_o$是权重矩阵，$x_t$,$h_t$是t时刻的输入状态和隐态，$\sigma$是 sigmoid函数，$\cdot$是按元素的乘积，文本的引导向量guidance vector是各个时间步隐态的算术平均值：  
$v_text=\frac{\sum^{i=1 \to \L}{h_t}}{L}  

### 3.4 Early Fusion 早期融合  
在文本分类任务中，Bi-LSTM的初始状态通常设置为零，但它是一个潜在的多模态信息输入点，可以促进模态对文本的理解。在该模型中，我们采用非线性变换的属性引导向量作为Bi-LSTM的初始状态。  
![img](https://github.com/Kittyuzu1207/Share/blob/master/img/04175.png)  
其中$h_{f_0}$，$c_{f_0}$是前向LSTM的初始状态，$h_{b_0}$，$c_{b_0}$是后向LSTM的初始状态，ReLU表示元素级别的ReLU激活函数，$W$和$b$是权重矩阵和偏差。  
我们同样也尝试了使用图象的引导向量来作早期融合（作LSTM的初始状态），但效果不好   

### 3.5 Representation Fusion 表征融合 
受到VQA视觉问答中attention机制的启发，表征融合旨在使用低维度的原始向量和高维度的引导向量来重构特征向量$v_image$,$v_attr$,$v_text$。低维度的原始向量即 文本模态中每个时间步t的隐态{$h_t$},图象模态中的196个rigional vectors，属性模态中的 attribute embeddings。  
我们使用${X_m}^i$来表示模态m的第i个原始向量。这里的关键问题是计算每个${X_m}^i$的权重。用这个向量加权平均后得到模态m的新的表征。  
为了使用尽可能多的信息，更准确地建模多个模态之间的关系，在计算每个模态原始向量的权重时，利用所有三种模态的信息，也就是考虑三个模态的引导向量。对于每个模态m的第i个原始向量，根绝不同的模态n的引导向量计算三个引导权重${α_mn}^(i)$。最终的重构权重是规范化后的引导权重的平均值。  
![img](https://github.com/Kittyuzu1207/Share/blob/master/img/04176.png)  
其中$m,n\in{text,image,attr}$ 表示模态，${α_mn}^(i)$是模态m的第i个原始向量在模态n引导下的引导权重，${α_mn}$包含了模态m的所有原始向量在模态n引导下的权重。${α_m}^(i)$是模态m的第i个向量最终的重构权重。$L_m$是序列{${X_m}^i$}的长度，$W_mn1$,$W_mn2$是权重矩阵，$b_mn1$,$b_mn2$是偏差bias。  

在表征融合之后，$v_image$,$v_attr$,$v_text$就表示重构后的模态向量，作为后面layer的input  

### 3.6 Modality Fusion 模态融合  
我们不是简单地将来自不同模式的特征向量串联concat起来形成一个更长的向量，而是受（Gu et al,2018a）的工作启发的模式融合。首先将每个模态的特征向量$v_m$ 转换为一个固定长度的向量${v_m}^'$,然后采用两层的前馈神经网络来计算每个模态m的attention权重，它将用来计算${v_m}^'$的加权平均。最终结果是一个固定长度的向量$v_fused$。  
![img](https://github.com/Kittyuzu1207/Share/blob/master/img/04177.png)  
其中m是某个模态，$\hat{α}$是包含$\hat{α_m}$的向量。$W_m1$,$W_m2$,$W_m3$是权重矩阵,$b_m1$,$b_m2$,$b_m3$是biases。  

### 3.7 Classification layer 分类层  
我们使用两层完全连接的神经网络作为分类层。隐藏层和输出层的激活函数分别为元素级ReLu函数和sigmoid函数。损失函数为交叉熵。  

## 4 Dataset and Preprocessing  
由于没有公开的数据集可用于评估多模态讽刺检测任务，我们构建了自己的数据集。我们收集和预处理我们的数据类似（Schifanella等人，2016）。我们收集包含图片和一些特殊标签（如讽刺等）的英文推文作为正面示例（即讽刺），收集带有图片但没有负面示例（即非讽刺）标签的英文推文。我们进一步清理数据如下。首先，我们将含有讽刺、讽刺、反讽、讽刺的推文作为常规词汇丢弃。我们还丢弃包含url的tweets，以避免引入其他信息。此外，我们丢弃经常与讽刺性推文一起出现的词语，从而可能表达讽刺，例如笑话、幽默和炫耀。我们将数据分为训练集、开发集和测试集，比例为80%:10%:10%。为了更准确地评估模型，我们手动检查开发集和测试集，以确保标签的准确性。表1列出了我们最终数据集的统计数据。  
![img](https://github.com/Kittyuzu1207/Share/blob/master/img/04178.png)  
对于预处理，我们首先用一个特定的符号<user>替换user mentions。然后，我们使用NLTK工具包分离单词、表情符号和标签。我们还将标签符号#与标签分开，并将大写字母替换为小写字母。最后，单词在训练集中只出现一次，而不是出现在训练集中而是出现在开发集或测试集中，这些单词被替换为某种符号<unk>  

## 5 Experiments
### 5.1 Training Details
- Pre-trained models：pre-trained ResNet model is available online，word embedding 用Glove  
- Fine tuning： Parameters ofthe pre-trainedResNet model 在训练中是固定的.Parameters of word and attribute embeddings 在训练中更新.
- Optimization：Adam optimizer
- Hyper-parameters：fusion时神经网络的隐层大小hidden layer size是其输入大小input size的一半。表2   
![img](https://github.com/Kittyuzu1207/Share/blob/master/img/04179.png)  

### 5.2 Comparison Results  
结果对比  
![img](https://github.com/Kittyuzu1207/Share/blob/master/img/041710.png)  
- Random:随机预测是否为反讽  
- Text(Bi-LSTM)：用Bi-LSTM对文本建模再用clf层预测  
- Text(CNN)：用CNN对文本建模  
- Image:使用ResNet的image vector  
- Attr：只用attribute features 
- Concat： (2) means concatenating text features and image features, while (3) means concatenating all text, image and attribute features.   
可见，仅基于图像或属性模态的模型表现不好，而基于文本和文本模态的模型表现得更好，说明了文本模态的重要作用。Concat（3）模型的性能优于Concat（2），因为添加属性作为一种新的模态实际上引入了图像的外部语义信息，并在模型无法提取有效的图像特征时提供帮助。我们提出的分层融合模型进一步提高了性能，达到了最新的分数，表明我们的融合模型更有效地利用了三种模式的特点。  
我们进一步在我们提出的模型和文本（Bi LSTM）、Concat（2）、Concat（3）模型之间应用符号测试。零假设是我们提出的模型并没有比每个基线模型表现更好。符号测试的统计数据见表4。所有显著性水平均小于0.05。因此，所有的零假设都被拒绝，我们提出的模型明显优于基线模型。  
![img](https://github.com/Kittyuzu1207/Share/blob/master/img/041711.png)  

### 5.3 Component Analysis of Our Model 
我们进一步评估了早期融合、表征融合以及早期融合中不同模态表征对最终性能的影响。评价结果见表5。
![img](https://github.com/Kittyuzu1207/Share/blob/master/img/041712.png)  
我们可以看到，去除早期融合会降低性能，这说明早期融合可以改善文本的表示。早期的属性表示融合比图像表示融合效果好，说明了文本表示和图像表示之间的差距。如果去除了表示融合，则性能也会下降，这说明表示融合是必要的，表示融合可以细化每个模态的特征表示。

## 6 Visualization Analysis
### 6.1 Running Examples  
![img](https://github.com/Kittyuzu1207/Share/blob/master/img/041713.png)  
图3展示了一些讽刺的例子，我们提出的模型能够正确地预测它们，而只有文本模态的模型却不能正确地标记它们。结果表明，在我们的模型中，图像和属性有助于讽刺检测。举个例子，一个有危险的铲球的图片和一个写着“不危险”的文字在例子（a）中表达了强烈的讽刺。“尊敬的客户”与示例（b）中的凌乱包裹以及“凌乱”属性相矛盾。没有图像，成功地检测这些讽刺的例子几乎是不可能的。只有文本情态的模型无法检测到例如（c）的讽刺，尽管单词so在示例（c）中重复了好几次。然而，通过图像和属性模式，我们提出的模型可以正确地检测这些推文中的讽刺。  

### 6.2 Attention Visualization
![img](https://github.com/Kittyuzu1207/Share/blob/master/img/041714.png)  
图4显示了在表示融合阶段的一些示例的注意力。我们的模型能够成功地将注意力集中在图像的适当部分、句子中的基本单词和重要属性上。例如，我们的模型更关注不悦的表情符号和文本中的“amazing”一词，而在示例（a）中更关注阴暗的天空，因此，由于这两种模态的不一致，这条推特被预测为讽刺推特。在示例（b）中，我们的模型关注文本中的“严肃”一词，并关注图片中与“丰盛早餐”相矛盾的简单一餐，揭示了这条微博应该是讽刺性的。在示例（c）中，“yum”一词、“meat”属性和图像中的食物表示tweet的讽刺含义。  

### 6.3 Error Analysis
![img](https://github.com/Kittyuzu1207/Share/blob/master/img/041715.png)  
图5显示了一个我们的模型未能正确标记的例子，在这个例子中，图片中的侮辱性手势与短语“谢谢”形成对比。然而，该模型无法获得这种手势是侮辱的常识。因此，这张照片的注意力不集中在侮辱性的手势上。此外，属性也不能揭示图片的侮辱性含义，因此我们的模型无法预测这条微博是讽刺性的。  

## 7 Conclusion and Future Work
本文提出了一种新的层次融合模型，充分利用图像、文本和图像属性三种模式来解决具有挑战性的多模态讽刺检测问题。评估结果证明了我们提出的模型的有效性和三种模式的有用性。在以后的工作中，我们将把其他的形式，如音频，纳入到讽刺检测任务中，我们还将研究如何在我们的模型中使用常识知识。  



