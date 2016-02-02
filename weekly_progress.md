#Weekly Progress for Winter 2016 Research

##Table of Contents

Each week is indexed by the date of Monday.

* [11 January 2016](#11jan)
* [18 January 2016](#18jan)
* [25 January 2016](#25jan)
* [1 February 2016](#1feb)

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

####Our Ideas
* Could we make the posterior dependent on our topic model by adding appropriate edges in a graphical model? This might provide an opportunity for interesting structure learning.

####Data
* We still have no data set.

####Engineering

####Questions
* @Eric:
    * Unresolved question from [last week](#25jan)
