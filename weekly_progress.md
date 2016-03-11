#Weekly Progress for Winter 2016 Research

##Table of Contents

Each week is indexed by the date of Monday.

* [11 January 2016](#11jan)
* [18 January 2016](#18jan)
* [25 January 2016](#25jan)
* [1 February 2016](#1feb)
* [8 February 2016](#8feb)
* [15 February 2016](#15feb)
* [29 February 2016/6 March 2016](#29feb)

##Progress Reports

###<a name='11jan'>11 January 2016</a>

####Previous Work
Some of the following bodies of literature may be useful to us.

* **Deep learning and compressive sensing literatures on non-convex optimization**. Both of those communities seem to have nice results both in theory and in practice that might translate well to RL.

* **Adaptive data analysis** work from Cynthia Dwork and Moritz Hardt. This could help with evaluation, especially since the pricing problem involves sparse data.

* **Active/adaptive selection of pairwise comparisons** for our population heterogeneity model. There is work in this area from Adam Kalai, Robert Nowak, and Michael Jordan.

* **Deep RL** stuff. I don't want to be a deep learning researcher, but they made it work with LSTMs, so I should understand it to some extent.

* The research on POMDPs could be relevant since we expect incomplete/partially observable state. We are not using MDPs, but maybe we can take inspiration from their treatment of partial observability.

* John Lipor recently published a paper in which he was dealing with a situation somewhat like our pricing scenario, where the actions are real-valued, and there are no ground-truth targets (i.e. we don't consider a human decision to be optimal). He said that he ended up using a heuristic method to address the problem because it was difficult to get good generalization bounds.

####Our Ideas
* We intend on using **non-convex, non-Markovian** formulations of the problem.

* Two possible application directions, both involving population heterogeneity in reinforcement learning and using the heterogeneity model to inform the decision of which action to take.
    * **Adaptive Structure Learning.** Have multiple regularizers, where the constants are a function of the heterogeneity model. The result is an adaptive sparsity assumption that molds to the data. I want to look carefully at different types of regularizers and different choices of online optimization algorithms. For this, we would most likely use the Nielsen data set. We want to receive a person, based on their attributes, place them in a population segment, then choose an action (a product rec) based on the segment.
        * This one is my favorite, but it seems ambitious and open-ended for a first project. I want to work with something much more concrete first.
    * **B2B Discount Pricing.** A company wants to offer another company a discount on a large purchase in exchange for less wait time on the payment. How do we choose the discount? How does the choice affect the state? The state might include expectations of other business that interact with you, or future expectations of the same company. The choice of discount should also depend on the company to which the sale is being made. 
        * I think that characterizing the state for this problem and the noisy oracle issue will be quite interesting challenges, and we have a well-characterized real-world scenario and a clear goal, which I think will be good for me.

* Realistically, our data points will be incomplete. Would be interesting to try to attack this in a principled manner. POMDPs may be relevant. This might be better to think about once I have more progress.

####Engineering
* If I have to implement a lot of algorithms from scratch, I will almost definitely use Python and, if necessary, C++ for obvious reasons.

* It looks like there are many well-maintained RL software libraries around, written mostly in C++ and Java. We could use these to at least run our initial experiments. I've bookmarked (or followed on github) all of the ones that look promising and haven't fallen out of use/repair.

* Should investigate **Flux for running experiments**. Tianpei seems to think highly of it, and it sounds quite nice. Don't think I will need it, but who knows.

####Questions
* @Eric: 
    * What did you mean by 'full control' for the pricing project?
        * Unresolved.
    * Can you send me the data sets and some description and background whenever you get a chance?
        * Answer: still waiting on permission from the company, but will send it once we have permission.
    * Can you send me the info on Jake's reading group?
        * Answer: I got the big email from Jake anyway.

* @Al: 
    * Could the adaptive structure learning project start as an adaptive extension of Josh Meng's work? I'd be interested in taking that direction.
        * Answer: There are multiple ways to make an adaptive extension of his work. Some of it is already adpative in some sense. Yes, we can do something like this.

###<a name='18jan'>18 January 2016</a>

####Previous Work
* I made a separate file for a reading list and added a bunch of potential readings to it, including the ones you all mentioned in your responses to last week's report as well as some of my own browsings.

* I read and took notes on a bunch of the Bayesian RL tutorial.

####Our Ideas
* Clarified that we are mostly interested in the sense of adaptivity in which the environment is actually changing. We have not yet determined if we will assume that the environment is changing independent of or dependent upon the agent's actions, although the latter scenario seems more likely.

* Earlier we were thinking about having the regularization constants be functions of the population heterogeneity model. What if we also had an epoch schedule and made the epoch schedule a function of the population model?

* Use low-rank matrix factorization (with maybe a spectral interpretation) or sparse coding for state representation? This seems a bit like low-hanging fruit in terms of the idea itself. My guess is that somebody (or many people) has already done this.

####Engineering
* I was pointed to the UCI ML Repository, which has a tone of free data sets, so we can mess around on those until we have the pricing data.

* It could also be interesting to write up some code that simulates a random process with the characteristics that we are assuming (certain rate of drift, etc.). That is the kind of thing that might be useful to a lot of people.We could release it to the public maybe.  
* I set up a git repo with all of our documents (reading list, paper reviews, weekly reports, eventually code and documentation). I moved the repo over to the lab's account.

* Speaking of documentation, I should think about using some nice documentation library for Python or C++. Would be a good learning experience and good for reproducability.

####Questions
* @Eric:
    * Unresolved question from [last week](#11jan).
    * Will we be able to publish if we use the company's data?

* @Both:
    * I am reading the Foundations and Trends Bayesian RL tutorial, and it says that the regret of a Bayesian multi-armed bandit grows with the number of rounds, which is counter-intuitive to me. Did I just misread what they were saying, or do I misunderstand MABs?
    * Could you all take a look at the reading list PDF I sent to you and help me narrow it down a bit? I'd be happy to resend the list if the email got lost in your inboxes.
    * What perspective and approach are we taking on the population heterogeneity model? Matrix factorization? Bayesian topic model? Both?

###<a name='25jan'>25 January 2016</a>

####Previous Work
* Read more of the Bayesian RL tutorial and marked up some pages with questions and comments, some general RL questions, some related to our project specifically.

* Started reading Elad Hazan's online convex optimization book. I am just in the convex optimization review chapter, but there's some stuff that wasn't covered in 611, so I need it. I am going back to the Boyd book for details when necessary.

* There's an active learning paper from Sanjoy Dasgupta that uses a heirarchical clustering (e.g. heirarchical k-means, so not the same type of heirarchical as Bayesian heirarchical models) of the data seen up to the current time step to select data points for oracle query. It's not exactly what we want to do, but it has some nice theoretical results, and it is related to the idea of informing our choice of action with a model of heterogeneity in the population.

* I am checking out a paper from my old lab about a cool augmentation to structured objective functions that is a bit like variational inference and posterior regularization.
    * This paper directly informs learning of a model with a 'non-linear [and possibly non-convex] energy function'. The energy function can be parameterized by the data (e.g. the topic distributions from a topic model) and gives some score to the choice of parameters, so it is quite flexible.
    * The meta-algorithm uses some special projection method to efficiently project the minimizing model parameterization into the feasible region. The projection method is tailored to a (supervised) graphical model scenario, which involves a specific type of feasible region for the parameters called a marginal polytope. That is not what we are working on, but the non-local energy function could be useful for us.

* To clarify, I added the section on pairwise comparisons to the reading list because the population model may at some point involve something like a kernel matrix, which we will want to populate as efficiently as possible.

* Based on my coversation with Eric, I refined the reading list a lot.
    * Each thematic section is now split into readings that will be practically relevant now, practically relevant later, or relevant for eventual theoretical analysis.
    * There are now about 10 papers on the immediate high priority list, mainly related to bandits, reinforcement learning, and online optimization.
    * Most of the papers in the 'practical later' sections fall into two categories:
        * Methods for the population heterogeneity model, most related to matrix optimization problems
        * Methods for using the population model to inform the learning of the pricing or customer selection model

####Our Ideas
* I had a conversation with Al about target tracking, and he gave a concrete example of tracking a car along a road. I am not sure if he intended it has a metaphor for tracking a mobile (i.e. non-stationary) parameterization of our model, but that's the idea that popped into my head.

* It seems highly likely that we will be doing some sort of online matrix factorization. 

* Eric and I discussed the idea of bandits with delayed reward (i.e. with a survival model plugged in). 
    * This scenario starts with the typical bandit scenario, but instead of being limited to a single, instantaneous reward, each action provides a (random or deterministic) reward for a random number of contiguous time steps after the action is played. An example here is customer acquisition and churn for a subscription-based service.
    * The survival period and the cumulative reward from a single played action are linked here by a (deterministic?) function, which may simplify the scenario to some extent.
    * This has immediate practical implications as well as an interesting and novel methodological and theoretical component. 
    * As I understand it, bandits are an easier problem computationally than reinforcement learning. It may be good to start with this problem and gradually add the additional computational challenges introduced by an RL scenario.
    * This idea is similar to active learning, but with a very interesting sample/query probability function, which is the expected survival time/subscription length or equivalently the expected cumulative reward.

####Data
* The pricing data set gives no contextual features about companies who are receiving the offers. This is not ideal. Maybe go with the Nielsen data set instead.

* Gathered a couple more websites that maintain collections of data sets.
    * One is a website of US gov. data sets. Its called 'data.gov'. 
    * The other is Kaggle's new (I think free) data set collection.

* There's a good possibility of going with Nielsen since we have it, and it has rich features for observations and actions. Eric is thinking of other data sets as well.

####Engineering
* No updates this week.

####Questions
* @Eric:
    * Does the Nielsen data set have data for the delayed reward scenario?

* @Both:
    * Questions about the Bayesian RL monograph if Eric and I don't cover them in our meeting on Friday.

###<a name='1feb'>1 February 2016</a>

####Previous Work
* Should eventually read about submodularity if we work with MDPs or semi-MDPs.

* I added the measure-transformed CCA paper to my reading list. This could be a great starting point for the heterogenenous population model. It has interesting theoretical properties, and, according to the paper, its not much more complicated to implement that regular linear CCA. I need to figure out what it uses for the SVD routine under the hood. Hopefully its TroppSVD.

* If we want to extend to an online scenario, should investigate Dean Foster's students' online CCA paper, although I don't yet know enough about Yaya's project to know if that's necessary. I started reading the paper a bit more in depth since I have the background from Golub and Zha now. Its much easier to follow.

* I read a bunch of Golub and Zha's 1992 paper on CCA. I didn't get to the actual computation section, but we probably don't want to use their method anyway since its from 1992. I was mostly interested in the problem formulation section since the paper I am reading now relies on Gloub and Zha's problem formulation without giving much detail. It offered multiple perspectives, which was quite helpful, although I still have a few questions on some proof details.

* I read about half of the Joulani paper on delayed rewards and discussed it with Kareem. Its pretty clear from the paper that we'll need to significantly modify our approach to address the core of the 'subscription-based service' problem. 
    * In the paper, they assume that no information/reward is received until the delaye feedback comes back all at once. Since we expect something like month payments, this does not fit our model.
    * They see delay as completely undesirable. We see delay as desirable, but we would prefer if a group with shorter expected subscription length tends to cut their subscription earlier so that we stop recruiting them and start recruiting the other group as soon as possible.

* This is a bit out of left field, but I was reading about sum-of-squares optimization last night.
    * It is related to convex algebraic geometry and semi-definite programming. Basically, the idea is to find the coefficient matrix of a convex quadratic of polynomial equations, as in the x in the equation x^TAx is actually a polynomial whose square somehow models our problem, and we want to find optimal A.
    * It has become popular in the robotics and control communities lately. I have a friend at U-M in the MechE department whose adviser wants to use it to put a highly non-convex 'safety' distribution over possible routes from one location to another on a physical map. 
    * It is super cool and seems like a powerful tool just in terms of offering more expressive models. I am wondering if it is relevant to us given our eventual goal of reinforcement learning, which as strong ties to control and robotics. This is probably something we'd look at much later, but I'd like to throw it out there now.

####Our Ideas
* Could we make the posterior in a Thompson sampling scenario dependent on our topic model by adding appropriate edges in a graphical model? This might provide an opportunity for interesting structure learning.

* Extend the measure-transformed CCA to an online scenario. This could be a very interesting problem on its own. Maybe we could use the Dean Foster paper.

* If we start looking at MDPs or semi-MDPs, should think about showing the submodularity of the Bellman equation updates. If we can show a submodular property to the dynamic programming problem, we can put nice error bounds on it.

* There are definitely multiple opportunities to apply the delayed rewards algorithms on Yaya's data set. I can think of a couple examples of actions with delayed but incremental feedback:
    * The spread of the virus is a delayed effect of the action of innoculation.
    * The extent of success of a treatment could be a delayed reward of the treatment.

* Met with Kareem and discussed the Joulani paper. He thinks that we can get better regret bounds by imposing the additional structure of our problem onto the scenario they propose. A naive attempt doesn't get us anywhere, but he thinks a good direction is to put a geometric distribution on subscription length.

####Data
* For now, I will probably just use simulated data sets or data sets from UCI ML repo. 
    * It should be pretty easy to find a data set for testing CCA.
    * It may be trickier to find something to test the delayed rewards algorithms, but I can just simulate data for now. 

* Still waiting on data from Eric. He still hasn't really said what kind of data we'd use for the delayed rewards problem, although I think there'd be opportunities for it in Yaya's data set.

####Engineering
* Started implementing code for bandits with delayed reward experiments. Right now, I have a first draft of the BOLD meta-algorithm written up in Python. 

* My next steps are to 
    1. implement UCB and Thompson sampling (probably with beta-Bernoulli) and plug them into BOLD for testing
    2. repeat with the other meta-algorithm mentioned in the Joulani paper
    3. implement Kareem's idea and start toying around with my own solutions until I find something that works.

* For Yaya's project, I'd like to have prototypes of the measure-transformed and online CCA before we get the project data. I think I can start implementation pretty soon as I am starting to understand CCA much better.

####Questions
* @Al:
    * You mentioned that there are only 20-30 people in Yaya's data set. Is that going to cause problems for a heterogeneous population model?
    * Or is our 'population' for which the CCA is intended more abstract than the literal population of human subjects?
    * Do we know anything about the structure of Yaya's data matrix? There seems to be some good literature on how to modify CCA computations to take advantage of matrix structure.

###<a name='8feb'>8 February 2016</a>

####Previous Work
* Sivaraman Balakrishnan has a paper [Sparse Additive Functional and Kernel CCA](http://www.stat.cmu.edu/~siva/Papers/CCA12.pdf) that I am looking at since we are thinking of using functional CCA for Yaya's project.

* I have finished going through the online CCA paper except for the proofs, which are not high priority right now. It mentions a couple pieces of previous work about using CCA to find correlations between genotype and phenotype, which I believe is part of what we want to do, so I am going to check those out as well.

* I have started working through the measure-transformed CCA paper. It is much less over my head than I expected, although it may take me a while to get through the whole thing.

* There's a cool paper from Jake's postdoc and some other interesting learning theory people on adaptively adding polynomial features to a regression problem. They just show results for linear regression. Maybe we could extend that to CCA, although this is basically a partial kernelization, which is what we are trying to avoid.

* Online CCA is basically an open problem. FLSLDSCCA's online stochastic algorithm has significantly lower performance than their batch algorithm. Also, I think they mention that the batch version can be extended to kernel CCA easily, but it requires the whole kernel matrix, which may counteract the performance benefits they get from their gradient-based algorithm. They don't offer experiments with a kernelized version.

####Our Ideas
* The paper [Finding Linear Structure in Large Datasets with Scalabe Canonical Correlation Analysis](http://arxiv.org/pdf/1506.08170.pdf) (henceforth referred to as FLSLDSCCA) has an interesting topic in the future work section. They claim that their algorithm enables easy thresholding where normal CCA would not. They also claim that this thresholding performs well empirically. They encourage further investigation here. Maybe this could be us?

* We could try to show that its easier to implement a non-linear online CCA via the measure transformation than a kernelization.

####Data
* Downloaded the AlgoSnap/Crowdsignals data and looked at the documentation a bit. I am going to try running it through my CCA code once that is ready.

* Wrote some code to produce random low-rank matrices.

* Going to write some code to simulate delayed reward data.

####Engineering
* I am developing a nice, modular Python framework for running our delayed feedback bandit experiments. It is very bare-bones right now, but I think it will get more sophisticated as we start testing more algorithms and running on more data sets.

* I have first draft implementation of the exp3 and ucb1 algorithms within the framework I've set up. My next step is to make a data simulator to test the implementations first on their own, and then in the BOLD meta-algorithm.

* CCA:
    * I have a Python implementation of the batch algorithm from the CCA paper mentioned in the 'Our Ideas' section. The code looks pretty good, but I need to actually run it on some simulated data to see how it compares with something like Scikit-Learn's CCA implementation. I also still need to add some of the things they mention in the experiments section, like some perturbation/regularization for stability.
    * I implemented some randomized linear algebra algorithms from [this paper](http://arxiv.org/abs/0909.4061). 
        * One is for finding an orthonormal basis, and the other is for finding an SVD. 
        * This should help with scaling up pretty much any of our ideas for CCA.
        * Right now, I am just using Numpy's QR and SVD implementations for subroutines underneath the randomized meta-algorithms.
    * Depending on the scale of our data or the slowness of Python, I may need to reimplement some of this stuff in C or C++.
        * There are good SVD implementations that I can use, but I'll have to change the wrappers that handle randomization.
        * This will be good practice with Cython and Eigen so that I am better prepared to implement more complex things later on.
    * I've implemented some short utility functions for tedious code snippets that I expect to use many times.
    * All of the code is organized into thematic modules, which is nice for code reuse and eventual release to the public.

* For the most practical considerations, I am looking into
    * Doxygen or Sphinx to make documentation for the code I am writing
    * nosetests for my testing framework
        * This will work for Python. I am not yet sure what I'll use for C++ code. I used cxxtest at my summer internship, so that's probably the first thing I'll check.
    * bokeh for charts and graphs
        * I used bokeh a bit last semester, and I saw some people use it over the summer. Its much prettier and more versatile than matplotlib and at least as easy to use.
    * Eigen for C++ linear algebra, and Cython for interfacing with Python
        * I have browsed some tutorials and documentaiton for these two, and it looks like its fairly common to combine them (this is not surprising), so its probably a good direction to take if we want to scale.

####Questions
* @Al:
    * I have a much better understanding of CCA in general, but I am a little unclear on the semantics of the data matrices for our specific application. What corresponds to each row and each column of each of the data matrices? What are the two sets of variables we are working with?


###<a name='15feb'>15 February 2016</a>

####Previous Work
* I just found a series of papers on deep estimators for CCA, including a paper on SGD-based optimization of a CCA objective. Should keep an eye on this so we don't repeat other work. Plus their stochastic optimization method may be useful.

* Eric gave me a few papers on bandits applied to marketing scenarios, one of which considers delayed rewards, and another of which considers population heterogeneity.

####Engineering
* CCA
    * Reimplemented some of the randomized linear algebra in C++ with considerable performance improvements.
    * Refactored randomized linear algebra for easier interaction with Python and future use.
    * Started a Makefile for compiling the shared library of the C++ code so I can easily import into Python.
    * Started writing Cython bridge to use C++ code on the Python side.

* Bandits
    * Beta-Bernoulli bandits, UCB1, and Exp3 are now all working correctly.
    * Bold has at least one remaining bug, but I have a very rudimentary test just for code correctness. Should be able to tease out any bugs this way.

####Our Ideas
* Meeting with Eric.
    * Discussed possible extension of BOLD and other solutions to address our specific problem.
        * Warm start each new learner with the wins and losses up until now.
            * Maybe warm-start with some sort of regularization?
        * Try just vanilla learners with internal table look-up for keeping track of delayed rewards.
        * Try bandits that maintain a probabilistic model for delay length.
    * Discussed test cases for simulated data-generating process. Probably will focus on stochastic data generating process.
        * Different distributions with different parameters.
        * Different spread among arm means resulting in different difficulties of differentiation between arms.
        * Fixed, constant delay time.

* Meeting with Al.
    * Discussed my questions and confusion about measure-transformed CCA.
    * Determined course of action for long-term online, non-stationary CCA.
        * Start digging into the Puterman book on MDPs.
        * Derive simplest possible probabilistic model for drifting data-generating distribution.
    * Determined short-term course of action for project with Yaya.
        * Finish implementing Zhuang Ma online CCA.
        * Implement measure-transform CCA.
        * Combine online and MTCCA to create online MTCCA.
            * Will need to try at least two schemes for learning the MT parameters.
                * Pre-compute parameters for MT functions.
                * With each round of mini-batch gradient ascent, use the mini-batch to make gradient updates to both the MT parameters and the CCA parameters.

####Data
* Yaya's data is coming in soon.

* We will probably use the Bonobos data for delayed rewards. 

####Questions
* None that I can think of right now.

###<a name='29feb'>29 February 2016 and 6 March 2016</a>

####Previous Work
* Hogwild could be quite helpful to us, especially considering our desire to impose sparsity penalties/constraints. The idea behind Hogwild seems quite similar to the idea behind the marginal likelihood for distributed parameter estimation paper, which will also be quite useful.

####Our Ideas
* Use differential geometry to do (functional?) CCA on some manifold so it is more sensitive to non-linear relationships. Is this somehow equivalent to the measure transform? This is sort of inspired by reading about Fisher info and natural gradient.

* Somewhat related to the previous bullet point, what do the canonical vectors tell us about the geometry of our problem? Does it make sense to use them as parameters for some sort of quadratic norm regularizer for a downstream optimization problem?

* I think SciKit-Learn's CCA implementation has a bug. Their canonical bases do not produce an identity when plugged into a quadratic parameterized by the empirical second moment matrix.

* I wonder if we could focus more on the distributed aspect of BOLD and its possible extensions. Could we find some unique advantage to those types of algorithms from this perspective?
    * Potential issues: BOLD may require a lot of communication overhead. Not sure, though. Should think about this carefully.
    * Maybe the new visiting researcher or Yasin will have some ideas about this?

* Imagine the following problem:
    1. A user specifies arbitrary dependence between rewards, delays, context, state, and actions.
    2. From the specification, we automatically produce an algorithm that gives good performance in the specified scenario.
    
    Wouldn't it be awesome if we could do this? I am thinking of something a bit like STAN, but for online learning.

* Active learning is usually sold as a way to minimize queries to expensive labeling oracles, but labeling is not necessarily the only cost associated with a specific training example. We could think about generalizing the idea of the cost of a training example, e.g. communication overhead of making parameter updates in a distributed system or computational cost of a model with expensive updates. Interesting aspects to this idea:
    * not specific to supervised learning problems
    * can possibly deal with multiple costs and actions (e.g. considering the costs of oracle query, update computation, and communication overhead, should we label this example, and to which nodes should we send the parameter update?)

* How hard would it be to develop distributed estimation algorithms that take advantage of structural sparsity _and_ are able to deal with adaptive structural assumptions?

####Meetings
* Al:
    * In general, we are interested in active learning and CCA. First we should focus on some particular challenges within CCA, then impose an active learning scenario onto that.
    * Lately has been more of a survey stage. Now we need to start zooming in. First focus on a problem formulation, then methodology, then scalable computation, then real-world applications.
        * **Possible problem formulations with possible methodologies.**
            * Non-linear CCA
                * Measure-transform functions
                    * For now, pre-compute measure-transform parameters, then plug into CCA. Later, simultaneously learn both.
                    * Need to find best way to learn measure-transform functions. For now just try SGD. Maybe get more sophisticated later.
                * Kernelization
                    * Can this be scaled? Maybe use random matrix approximations?
            * Time-varying CCA
                * Need to set up a state space model and formulate the dynamic programming problem.
            * Graph CCA
                * Start with a graph where each node is an instance (with internal structure between multiple modalities represented by a graphical model), and each edge represents dependency that results in a consistency constraint. End up solving an SDP-type problem for a non-convex objective function. Should investigate John Wright papers to see what kind of structural penalties we can impose on our objective to result in well-behaving non-convexity.
            * Asynch CCA
                * If we choose time invariance, this can definitely be done with something like Hogwild, which also appeals to potential sparsity assumptions.
                * Can we do time-varying CCA with asynchronous, stochastic updates?
                    * Split the time series into blocks to be used for the asynchronous updates. Will blocks of size two be enough, or do they need to be bigger? Is there some way to characterize a trade-off between block size, sample complexity, and empirical performance?
                * For the active learning component, the asynchronous algorithms used for delayed reward could be applied, or we could take inspiration from them somehow.
        * **Possible applications.** 
            * For these applications, there is available data, and we already have collaborators, both domain-specific and in engineering.
                * Web search: allow more interesting search queries by things like similar image or text content.
                * Marketing: want to take population non-homogeneity into consideration when making marketing decisions like who to target and how to best reach them.
                * Events in Twitter: I am a bit skeptical about the ability to gain actionable information from individual or even sequences of tweets, but we are thinking more of very large aggregations, so it may not be necessary to master the NLP aspect.
                * Genomic: continue working with dataset from Yaya.
                * Astronomy: discovery of dark matter through multiple measurement types.
        * **Scalability and Distributed Computing.** 
            * As disparity between ambient and true dimension goes up, projected gradient becomes slower.
            * Make graphical model structural assumptions to facilitate distributability of the computation.
            * I will need to start familiarizing myself with distributed estimation techniques, on both theoretical and engineering levels. Thibaut also seems interested in this topic.
            * Most of the decentralized estimation techniques I have read about involve subgradient methods, which are very slow.

####Engineering
* CCA:
    * The Cython layer for the randomized SVD tool is fixed, and I have a _much_ better general understanding of Cython now. Using it from this point forward should be much easier.
    * There is an odd bug in the randomized SVD so that all of the corresponding entries of my singular vectors and numpy's singular vectors have identical absolute value, but some of the signs are flipped, and there doesn't seem to be a pattern to the sign flips. Is this a numerical stability issue?
    * Implemented the stochastic gradient CCA in Python.

* Delayed Rewards:
    * Made a ton of progress on a modular data-serving framework that can include data from pretty much any source: static or streamed, simulated or real.
    * Started running tests with this framework to find that, when plugged into BOLD, exp3 and ucb1 seem to be making random decision even after ~1000 rounds on 6 arms of Bernoulli rewards with means 0.1, 0.1, 0.1, 0.1, 0.1 and 0.5, and a fixed delay of 1 for each arm. Need to investigate what's wrong with my implementation.

####Questions
* @Both:
    * Are there scenarios in which the cost of computing a parameter update is dependent on the observation?
