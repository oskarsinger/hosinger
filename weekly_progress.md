#Weekly Progress for Winter 2016 Research

##Table of Contents

Each week is indexed by the date of Monday.

* [11 January 2016](#11jan)
* [18 January 2016](#18jan)

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
* Made a separate file for a reading list.

####Our Ideas
* Clarified that we are mostly interested in the sense of adaptivity in which the environment is actually changing. We have not yet determined if we will assume that the environment is changing independent of or dependent upon the agent's actions, although the latter scenario seems more likely.

* Earlier we were thinking about having the regularization constants be functions of the population heterogeneity model. What if we also had an epoch schedule and made the epoch schedule a function of the population model?

* Use low-rank matrix factorization (with maybe a spectral interpretation) or sparse coding for state representation? This seems a bit like low-hanging fruit. My guess is that somebody (or many people) has already done this.

####Engineering
* I was pointed to the UCI ML Repository, which has a tone of free data sets, so we can mess around on those until we have the pricing data.

* It could also be interesting to write up some code that simulates a random process with the characteristics that we are assuming (certain rate of drift, etc.). That is the kind of thing that might be useful to a lot of people.

* I set up a git repo with all of our documents. It will also eventually hold code and documentation for the code.

* Speaking of documentation, I should think about using some nice documentation library for Python or C++. Would be a good learning experience and good for reproducability.

####Questions
* @Eric:
    * Unresolved question from [last week](#11jan).
    * Will we be able to publish if we use the company's data?

* @Both:
    * I am reading the Foundations and Trends Bayesian RL tutorial, and it says that the regret of a Bayesian multi-armed bandit grows with the number of rounds, which is counter-intuitive to me. Did I just misread what they were saying, or do I misunderstand MABs?
