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


