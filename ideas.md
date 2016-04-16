#Ideas for My RL Project

* * *

##Core ML

###Bregman Divergence CCA

####Motivation
* More flexible than normal CCA

####Approach
* Probably just start by checking out Mahalanobis distance

* * *

###Adaptive Epoch Length
Split the learning into epochs and make the epoch length a function of the data (or the data's heterogeneity model).

####Motivation
* Might address the issue of a non-stationary distribution.

####Approach
* Maybe it could come from some operator norm evaluation on our low-rank matrix factorization?

* * *

###Adaptive Regularization
Make the regularization constants a function of the data (or the data's heterogeneity model).

####Motivation
* This could result in regularization that adapts as the population's structure changes.

####Approach
* Maybe it could come from some operator norm evaluation on our low-rank matrix factorization?

* * *

###Heterogeneous Population Model
* Attempt to account for different subgroups of the population in a data-dependent manner.

* Directly throw a matrix norm into our main objective function as a regularizer/non-local energy function. Which matrix norm? What do we want to encourage with the norm?

####Motivation
* Could result in interesting low-dimensional state representation.

####Potential Approaches
* Heirarchical Bayesian model
* Low-rank matrix factorization

* * *

###Low-D State Representations
Develop a method for compressing state representations.

####Motivation
We are almost definitely going to have high-D state, so we will need to compress it somehow. Deep learning has been successful at producing salient, concise state representations, but it has its own limitations. It would be nice if we could come up with something that was more efficient to learn or required less data, maybe with a small loss in accuracy.

This idea seems like a more general version of the population heterogeneity model if I am understanding the latter correctly.

####Approaches
* Low-rank matrix factorization
* Sparse coding

* * *

##Applications

* Look around for people in other fields who are interested in or already using RL. Places to start:
    * MIDAS faculty
    * Engineering faculty
    * Natural science faculty

* Laura pointed me to the [UCI ML Repository](http://archive.ics.uci.edu/ml/index.html), which has a ton of cool data sets. Looking at the time series data sets to see if we can work with any of them.

###Telescope Measurements
* Should talk to Paul about the astronomy/telescopes measurement thing. That would be quite an interesting project.

* We could try to determine when/whether to schedule a telescope session, or even where in the sky to take measurements.

* Potential issues include
    * sparse (or non-existent) data on measurement days
    * difficult data format like some sort of weird image

###Clinical Stuff
* Keep in touch with Ambuj Tewari and keep looking at Susan Murphy's page, especially since Ambuj said that he and Susan would be teaching that RL-for-health course next year.

* The Hulu/Netflix binge-watching data set that Eric mentioned is a really interesting example of addiction behavior. Feels a little sinister, though. Think carefully about what your project's objective is, who would be using your work, and what they are gaining from it. Think about how you would feel if you knew somebody else was using your data this way, even without your name stamped on it.
