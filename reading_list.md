#Tentative Reading List for Winter 2016 Research

* This is organized vaguely thematically and also vaguely by author. May try to refine it later.
* Each section is split into papers with more immediate practical relevance and things that will be more useful down the line when we need to do rigorous theoretical analysis, or make more sophisticated extensions to our methodology.
* I should decide on some markings that indicate which part of the project motivates each paper, e.g. population heterogeneity, model updates and optimization, structural regularization, etc.

* * *

##Machine learning.

* * *

###Structure learning and sparsity in graphical models. 
This is mostly from Josh Meng.

####Practical later.
* **Learning Latent Variable Gaussian Graphical Models** ([arXiv](http://arxiv.org/abs/1406.2721))
    * I think that understanding this paper might be important for approaching the problem of a heterogeneous population.
* **Marginal Likelihoods for Distributed Parameter Estimation of Gaussian Graphical Models** ([arXiv](http://arxiv.org/abs/1303.4756))
* **Distributed Learning of Gaussian Graphical Models via Marginal Likelihoods** ([pdf](http://web.eecs.umich.edu/~mengzs/meng_aistats13.pdf))

####Theoretical.
* **Understanding the Limiting Factors of Topic Modeling via Posterior Contraction Analysis** ([pdf](http://web.eecs.umich.edu/~mengzs/lda_icml2014.pdf))

* * *

###Pairwise comparisons.
This is mostly from Kevin Jamieson and Robert Nowak.

####Practical later.
* **Active Ranking using Pairwise Comparisons** ([pdf](http://www.cs.berkeley.edu/~kjamieson/resources/activeRanking_extended.pdf))
* **Low-Dimensional Embedding using Adaptively Selected Ordinal Data** ([pdf](http://www.cs.berkeley.edu/~kjamieson/resources/activeMDS.pdf))
    * This could be useful for developing efficient state representations from the population heterogeneity model.
* **Efficient Ranking from Pairwise Comparisons** ([pdf](http://www.cs.berkeley.edu/~jordan/papers/wauthier-jordan-jojic-icml13.pdf))

####Theoretical.
* **Query Complexity of Derivative-Free Optimization** ([pdf](http://www.cs.berkeley.edu/~kjamieson/resources/QueryComplexityOfDFO.pdf))

* * *

###Reinforcement learning and bandits.

####Practical now.
* **Bayesian Reinforcement Learning - A Survey** 
    * This one's behind a pay wall, but you can get it through U-M's library subscription to Foundations and Trends. Also, I have a copy...
* **On-the-Job Learning with Bayesian Decision Theory** ([pdf](http://cs.stanford.edu/~pliang/papers/onthejob-nips2015.pdf))
    * This doesn't explicitly use the words 'reinforcement learning' (though they do mention policies and Monte Carlo Tree Search) or 'active learning' in the abstract, but it sounds like that's what it is. It seems like they have both nice theory and serious empirical results in the context of a real-time, deployed system, which is super cool.
* **Online Learning under Delayed Feedback** ([pdf](http://jmlr.org/proceedings/papers/v28/joulani13.pdf))

####Practical later.
* **Bayesian Nonparametric Bandits** ([project euclid page](http://projecteuclid.org/euclid.aos/1176349753))

####Not sure.
* **Optimal Sensor Scheduling via Classification Reduction of Policy Search (CROPS)** ([pdf](http://web.eecs.umich.edu/~hero/Preprints/BlattHero_ICAPS06.pdf))
    * Home team!
* **From Weighted Classification to Policy Search** ([pdf](http://papers.nips.cc/paper/2778-from-weighted-classification-to-policy-search.pdf))
    * Home team!
* **Large-Scale Bandit Problems and KWIK Learning** ([pdf](http://web.eecs.umich.edu/~amkareem/pubs/AbernethyAminDraiefKearnsICML2013.pdf))
    * This covers an interesting scenario of a large (and possibly infinite, although not continuous) action space with a growing number of 'available' actions. Also, its by Jake and Kareem!

* * *

###Deep reinforcement learning.

####Eventually.
This is mostly from Deep Mind. There are at least 10 deep RL pubs from them in the last year. I don't yet know enough about RL to choose which ones look promising. Most of them seem to be focused on using deep learning to compress high-dimensional state spaces into something more manageable using a deep model. I didn't read deeply (ouch, that's cheesy) enough to understand whether they were using supervised or unsupervised deep models or doing pre-training.

* * *

###Markov decision processes.
* **Dynamic Probabilistic Systems, Volume II: Semi-Markov and Decision Processes. Vol. 2. Courier Corporation, 2013.** ([book](http://store.doverpublications.com/0486458725.html))
    * Need to find a pdf or a cheap used hard-cover. Maybe Al has a copy lying around.
* **Markov Decision Processes: Discrete Stochastic Dynamic Programming**
    * Found a pdf, would be nice to find a cheap used hard-cover. Maybe Al has an extra copy lying around.
* **Dynamic Allocation of Pharmaceutical Detailing and Sampling for Long-Term Profitability** ([pdf](http://www.dii.uchile.cl/~rmontoya/papers/Dynamic_Allocation.pdf))

* * *

##Optimization.

* * *

###Proximal Methods, Mirror Descent, FTPRL, etc.

####Practical now.
* **On the importance of initialization and momentum in deep learning** ([pdf](http://www.cs.toronto.edu/~jmartens/docs/Momentum_Deep.pdf))
    * This paper covers some pretty crucial aspects of optimizing deep neural networks, especially with respect to the difficulties encountered because of non-convexity.
* **New insights and perspectives on the natural gradient method** ([arXiv](http://arxiv.org/abs/1412.1193))
    * This one comes from the deep learning community, and its quite recent. It has a nice theoretical interpretation (related to approximate Hessians and the conditioning of the problem), it is efficient, and hopefully it is empirically effective.
* **Adaptive Subgradient Methods for Online Learning and Stochastic Optimization** ([pdf](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf))
* **Composite-Objective Mirror Descent** ([pdf](http://www.cs.utexas.edu/users/ambuj/research/duchi10composite.pdf))
* **A Unified View of Regularized Dual Averaging and Mirror Descent with Implicit Updates** ([pdf](http://arxiv.org/pdf/1009.3240.pdf))
* **Stochastic Methods for l1-regulared Loss Minimization** ([pdf](http://www.jmlr.org/papers/volume12/shalev-shwartz11a/shalev-shwartz11a.pdf))
* **Mirror descent and nonlinear projected subgradient methods for convex optimization** ([pdf](https://web.iem.technion.ac.il/images/user-files/becka/papers/3.pdf))

* * *

###Regularization.

####Practical later.
* **Linear System Identification via Atomic Norm Regularization** ([pdf](http://www.eecs.berkeley.edu/~brecht/papers/12.Sha.EtAl.Hankel.pdf))
    * I think this paper might be useful for choosing (or adapting) the regularizer that encodes our structural assumptions.
    * This paper seems particularly relevant given the connections between control and RL, but the other atomic norm stuff is pretty cool too.
* **Practical Large-Scale Optimization for Max-Norm Regularization** ([pdf](http://www.eecs.berkeley.edu/~brecht/papers/maxnorm.NIPS10.pdf))
    * Similar to previous one, could be helpful when we are picking our regularization scheme, particular since we are thinking about topic models, which can be viewed as low-rank matrix factorization.
* **Bethe Projections for Non-Local Inference** ([arXiv](http://arxiv.org/abs/1503.01397))
    * This is a sort of prediction-time regularization-type-thing, but its a little more sophisticated. Going to read a bit more to understand whether its relevant for us. The first authors are friends/research mentors from my old lab!

####Theoretical.
* **Fighting Bandits with a New Kind of Smoothness** ([pdf](http://papers.nips.cc/paper/6030-fighting-bandits-with-a-new-kind-of-smoothness.pdf))
    * The focus of the paper is a new regularization scheme. Not really sure about its relevance to us, especially since it focuses on convex scenarios, but I am curious, and maybe we will get something out of it.
    * Is the name of this paper a cheeky reference to Stephen Wolfram's book?

* * *

###Distributed.

####Practical now.
* **Taming the Wild: A Unified Analysis of Hogwild!-Style Algorithms** ([arXiv](http://arxiv.org/abs/1506.06438))
* **Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent** ([pdf](https://www.eecs.berkeley.edu/~brecht/papers/hogwildTR.pdf))
* **Delay-Tolerant Algorithms for Asynchronous Distributed Online Learning** ([pdf](http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43138.pdf))

####Practical later.
* **DimWitted: A Study of Main Memory Statistical Analysis** ([arXiv](http://arxiv.org/pdf/1403.7550))
* **Exponential Stochastic Cellular Automata for Massively Parallel Inference** ([pdf](http://www.manzil.ml/res/Papers/2015_NIPSW_sca.pdf))

* * *

###Numerical linear algebra.

####Practical later.
* **Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions** ([arXiv](http://arxiv.org/pdf/0909.4061v2))
* **Improved Matrix Algorithm via the Subsampled Randomized Hadamard Transform** ([arXiv](http://arxiv.org/pdf/1204.0062))
* **Low rank approximation and decomposition of large matrices using error correcting codes** ([arXiv](http://arxiv.org/pdf/1512.09156))

###Misc.

####Practical now.
* **Fast gradient descent for drifting least squares regression, with application to bandits** ([pdf](http://arxiv.org/pdf/1307.3176v4.pdf))
    * The abstract mentions something about an adaptive regularizer, and they speak about it in some detail later in the paper.
    * We are not doing least squares, but they show that SGD accounts to some extent for drifting target hypothesis (or something like that), and this could be helpful for us.
* **Introduction to Online Convex Optimization** ([pdf](http://ocobook.cs.princeton.edu/OCObook.pdf))
    * Much of the current non-convex optimization literature builds on online convex optimization, so I think it will be good to understand this stuff. 
    * Found a friend from Corso's lab who also needs to learn this stuff and wants to read it with me.
* **Global Convergence of Stochastic Gradient Descent for Some Non-convex Matrix Problems** ([arXiv](http://arxiv.org/pdf/1411.1134))

####Practical later.
* **NIPS 2015 Workshop on Non-convex Optimization for Machine Learning: Theory and Practice** ([website](https://sites.google.com/site/nips2015nonconvexoptimization/papers))
    * A lot of these may be batch-style algorithms, but some of them are definitely online.

####Theoretical.
* **Train faster, generalize better: Stability of stochastic gradient descent** ([arXiv](http://arxiv.org/abs/1509.01240))
    * Not from the deep learning community, but somewhat motivated by deep learning.
    * They show that running SGD for a few epochs over the full data set improves generalization, even for non-convex problems. Emphasis on a few. They note that running fewer epochs prevents overfitting, which makes intuitive sense as well.
    * They also show that some of the deep learning community's regularization heuristics and the practice of 'drop-out' improve their generalization bounds.

* * *

##Statistics.

* * *

###CCA.

####Practical now.
* **Finding Linear Structure in Large Datasets with Scalable Canonical Correlation Analysis** ([arXiv](http://arxiv.org/abs/1506.08170))
    * Offers efficient approximate algorithms for both batch and online CCA, both with nice theoretical guarantees.
    * Give experiments showing similar performance to un-approximated CCA on the same datasets.
* **On Measure Transformed Canonical Correlation Analysis** ([arXiv](http://arxiv.org/abs/1111.6308))
    * Extends CCA to non-linear relationships between variables with an easily-implementable modification to the join distribution.
    * Home team!
* **A greedy approach to sparse canonical correlation analysis** ([arXiv](http://arxiv.org/pdf/0801.2748v1))
* **Efficient Algorithms for Large-scale Generalized Eigenvector Computation and Canonical Correlation Analysis** ([arXiv](http://arxiv.org/abs/1604.03930))

###Bayesian models.

####Practical now.
* **Model Selection Using Database Characteristics: Developing a Classification Tree for Longitudinal Incidence Data** ([pdf](http://pubsonline.informs.org/doi/pdf/10.1287/mksc.2013.0825))
    * Home team!
* **Customer-Base Valuation in a Contractual Setting: The Perils of Ignoring Heterogeneity** ([pdf](http://www.brucehardie.com/papers/022/fader_hardie_mksc_10.pdf))

####Theoretical.
* **Risk and Regret of Heirarchical Bayesian Learners** ([pdf](http://arxiv.org/pdf/1505.04984.pdf))

* * *

###Adaptive data analysis.
This is mostly from Moritz Hardt

####Practical later.
* **Generalization in Adaptive Data Analysis and Holdout Reuse** ([arXiv](http://arxiv.org/abs/1506.02629))
* **Workshop on Adaptive Data Analysis** ([website](http://wadapt.org/index.html))
    * Should check out some of the papers from this workshop. They look like they might be important for understanding adaptive scenarios in general.
