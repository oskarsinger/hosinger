#Tentative Reading List for Winter 2016 Research

* This is organized vaguely thematically and also vaguely by author. May try to refine it later.
* This list is going for high recall, possibly low precision. These are the papers I think I might need to read. As the project evolves, I will almost definitelyremove some of them from the list.

* * *

###Structure learning and sparsity in graphical models. 
This is mostly from Josh Meng.

* **Learning Latent Variable Gaussian Graphical Models** ([arXiv](http://arxiv.org/abs/1406.2721))
    * I think that understanding this paper might be important for approaching the problem of a heterogeneous population.
* **Understanding the Limiting Factors of Topic Modeling via Posterior Contraction Analysis** ([pdf](http://web.eecs.umich.edu/~mengzs/lda_icml2014.pdf))
* **Marginal Likelihoods for Distributed Parameter Estimation of Gaussian Graphical Models** ([arXiv](http://arxiv.org/abs/1303.4756))
* **Distributed Learning of Gaussian Graphical Models via Marginal Likelihoods** ([pdf](http://web.eecs.umich.edu/~mengzs/meng_aistats13.pdf))

* * *

###Adaptive data analysis.
This is mostly from Moritz Hardt

* **Generalization in Adaptive Data Analysis and Holdout Reuse** ([arXiv](http://arxiv.org/abs/1506.02629))
* **Workshop on Adaptive Data Analysis** ([website](http://wadapt.org/index.html))
    * Should check out some of the papers from this workshop. They look like they might be important for understanding adaptive scenarios in general.

* * *

###Pairwise comparisons.
This is mostly from Kevin Jamieson and Robert Nowak.

* **Query Complexity of Derivative-Free Optimization** ([pdf](http://www.cs.berkeley.edu/~kjamieson/resources/QueryComplexityOfDFO.pdf))
* **Active Ranking using Pairwise Comparisons** ([pdf](http://www.cs.berkeley.edu/~kjamieson/resources/activeRanking_extended.pdf))
* **Low-Dimensional Embedding using Adaptively Selected Ordinal Data** ([pdf](http://www.cs.berkeley.edu/~kjamieson/resources/activeMDS.pdf))
* **Efficient Ranking from Pairwise Comparisons** ([pdf](http://www.cs.berkeley.edu/~jordan/papers/wauthier-jordan-jojic-icml13.pdf))

* * *

###Deep reinforcement learning.
This is mostly from Deep Mind. There are at least 10 deep RL pubs from them in the last year. I don't yet know enough about RL to choose which ones look promising.

* * *

###Online gradient-based optimization.
* **Fast gradient descent for drifting least squares regression, with application to bandits** ([pdf](http://arxiv.org/pdf/1307.3176v4.pdf))
    * The abstract mentions something about an adaptive regularizer. 
    * They show that SGD accounts to some extent for drifting target hypothesis (or something like that).
* **Train faster, generalize better: Stability of stochastic gradient descent** ([arXiv](http://arxiv.org/abs/1509.01240))
* **Introduction to Online Convex Optimization** ([pdf](http://ocobook.cs.princeton.edu/OCObook.pdf))
    * Much of the current non-convex optimization literature builds on online convex optimization, so I think it will be good to understand this stuff. 
    * Found a friend from Corso's lab who also needs to learn this stuff and wants to read it with me.
* **On the importance of initialization and momentum in deep learning** ([pdf](http://www.cs.toronto.edu/~jmartens/docs/Momentum_Deep.pdf))
* **New insights and perspectives on the natural gradient method** ([arXiv](http://arxiv.org/abs/1412.1193))

* * *

###Markov Decision Processes.
* **Dynamic Probabilistic Systems, Volume II: Semi-Markov and Decision Processes. Vol. 2. Courier Corporation, 2013.** ([book](http://store.doverpublications.com/0486458725.html))
    * Need to find a pdf or a cheap used hard-cover.
* **Markov Decision Processes: Discrete Stochastic Dynamic Programming**
    * Found a pdf, would be nice to find a cheap used hard-cover.

* * *

###Regularization.
* **Linear System Identification via Atomic Norm Regularization** ([pdf](http://www.eecs.berkeley.edu/~brecht/papers/12.Sha.EtAl.Hankel.pdf))
    * I think this paper might be useful for choosing (or adapting) the regularizer that encodes our structural assumptions.
    * This paper seems particularly relevant given the connections between control and RL, but the other atomic norm stuff is pretty cool too.
* **Practical Large-Scale Optimization for Max-Norm Regularization** ([pdf](http://www.eecs.berkeley.edu/~brecht/papers/maxnorm.NIPS10.pdf))
    * Similar to previous one, could be helpful when we are picking our regularization scheme, particular since we are thinking about topic models, which can be viewed as low-rank matrix factorization.
