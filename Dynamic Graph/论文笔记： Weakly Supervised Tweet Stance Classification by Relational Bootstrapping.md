## 论文笔记 Weakly Supervised Tweet Stance Classification by Relational Bootstrapping

### 1.方法 or 模型要点

#### 1.Task & Problem Formulation

stance classification: whether the author of the text is in favor of, against, or neutral towards a target of interest.

其他相关任务：user preference modeling； stance classification；

思路：from past posting behaviors+ friends' stance on issues

##### Task

Statistical Relational Learning (SRL) for stance classification. 

**STEP1** start from a small set of stance-indicative patterns and label the tweets as positive and negative

**STEP2** relational learner uses these noisy-labeled tweets to classify other...(hinge-loss)

ideas: a person is pro/against if writes a tweet.../ friends often agree.../ similar tweets-->similar stances

与情感分类的区别：

文本中可能没有明确提及target，也可能不是文本中的target

#### 2.Model: Stance Classification on Twitter

##### Markov Random Fields

<img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200912171535138.png" alt="image-20200912171535138" style="zoom:60%;" />

##### HL-MRFs for Tweet Stance Classification

MAP is NP-hard , difficult discrete optimization problem

**HL-MRFs** : allows for convex inference 

each potential function is a hinge-loss function

instead of discrete variables, MAP inference is performed over relaxed **continuous variables **with domain $[0, 1]^n$. These hinge-loss functions, multiplied by the corresponding model parameters (weights), act as penalizers for **soft linear constraints** in the graphical model.

补充：**hinge-loss** 

<img src="https://pic1.zhimg.com/80/v2-3c6aa9626ee8e4609b0d7c5712baf624_1440w.jpg?source=1940ef5c" alt="img" style="zoom:50%;" />



<img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200912221204125.png" alt="image-20200912221204125" style="zoom:50%;" />

<img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200912221728879.png" alt="image-20200912221728879" style="zoom:67%;" />

正则化项，类似于SVM中的margin

- 对于user和其twitter

  ​	势函数（potential function): $max(0,t_{ik}-u_{jk})$  , $t_{ik}$ 是tweet i has label k, $u_{jk}$ 是tweet j has 	label k.

  ​	（此函数用于测量用户及其tweet的不同标签的惩罚。）

- 对于user和其friends

  potential funtion: $max(0,u_{ik}-u_{jk})$ 

- tweet-tweet

  $s_{ij}max(0,t_{ik}-t_{jk})$

  其中$s_{ij}$ 是tweet i,j 之间的similarity(This scalar helps penalize violations in proportion to the similarity between the tweets)  cosine simi : 1-4 gram, 阈值0.7

- two hard linear constraints are added,

  $\sum_k t_{ik}=1$ ,    $\sum_k u_{ik}=1$

Weight learning is performed by an improved structured voted perceptron:

<img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200912224531316.png" alt="image-20200912224531316" style="zoom:67%;" />

idea: 这只是一部分规则，实际上还可以考虑mention，retweet network，textual entailment

#### 3.Experiments

##### Data

SemEval-2016， 78，000+ tweets about "Trump", allowed minimal manual labeling

<img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200912225018716.png" alt="image-20200912225018716" style="zoom:67%;" />

goal: test stance towards the target in 707 tweets

two-phrase approach: 1. predict the stance of the training tweets using our HL-MRF. 2.we use the labeled instances as training for a linear text classifier.  [Algorithm Relational Bootstrapping.]

initial positive and negative instances:  have at least one positive or one negative pattern, and do not have both positive and negative patterns;

<img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200912225958952.png" alt="image-20200912225958952" style="zoom:80%;" />

seeds太少，self-learning poor-->augment the dataset with tweets that our relational model classifies as positive or negative with a minimum confidence (class value 0.52 for pro-Trump and 0.56 for anti-Trump). 

##### Classification

linear-kernel SVM: N-gram, Lexicon, Sentiment 

#### 4. Case Study

Demographics of the Users

<img src="C:\Users\YuzuK\AppData\Roaming\Typora\typora-user-images\image-20200912230302284.png" alt="image-20200912230302284" style="zoom:50%;" />



### 2.写作要点

### 词&短语：

electoral issues； a wealth of information；

### 句：