#Engineering

##Tests

###Correctness
* Write up some unit tests in nosetest for
    * CCA
    * SVD
    * QR
    * linal utility functions
    * exp3
    * ucb1
    * tsbb

###Empirical Performance
* Write up a somewhat general framework for running experiments.
    * Would be nice if it automatically printed charts/graphs/tables.

####Randomized Linal

####CCA
* Test CCA against Scikit-Learn's implementations.
    * Issue: the canonical basis returned by sklearn does not satisfy the quadratic norm orthonormality constraints of the CCA problem.
* Test CCA against Matlab's implementation.
* Figure out why CCA is unstable. For some reason, the initial first one or two singular values of the normalization and Gram matrices are very large, both absolutely and relative to the others, so the gradients and singular values in subsequent rounds are blowing up. Taking 90% of the energy from the singular values doesn't seem to be helping, so I need to find another solution.

####Bandits
* Test against various data scenarios:
    * Pre-set adversarial with varying levels of difficulty. 
        * Is there a 'smooth' function for tuning difficulty of an adversary?
    * Online adversarial with varying levels of difficulty.
        * Is there a 'smooth' function for tuning difficulty of an adversary?
    * Subroutines without BOLD, just one learner with cumulative rewards, where action corresponding to reward is visible to learner.
    * Warm start within BOLD.
    * Updating as we are waiting (censored observations).
    * Different decay rates in the geometric parameters.
    * Different scales of geoemtric parameters.
    * Draw ps from a beta. Different alpha and beta parameters would result in different decay rates.

###Efficiency
* Start doing stress tests on the scale that can be handled by the Python code.
    * This is especially relevant for CCA since computation for bandit algorithms is not super intense.
    * Do time trials to justify randomized linear algebra algorithms. Its getting faster (for low-rank matrices)!

* Do some more thorough benchmarking on the randomized linear algebra.
    * Probably need to find a decent benchmarking library.

##Code Structure

###Optimization
* Figure out how to elegantly have my optimization algs deal with both Stiefel/Grassman and Rn. Do I need to make separate objects for the matrix ones, or can I cleanly introduce the logical split in the existing ones? Probably separate objects will be much easier, especially since their is extra state to maintain for the matrix versions.

###Bandit Framework

###CCA
* Should I try to put any framework-y type structure on the CCA stuff?
* Make sure to add more diagnostics in CCA code like printing the current objective value, etc.
    * Should add a "verbose" mode to enable such diagnostics.
    * Should consider logging as well?

###C++/Python Interaction

##Algorithms

###Optimization
* Read that Edelman paper to hopefully figure out how to do quasi-Newton methods on matrices without a four-mode tensor.

###Randomized Linear Algebra
* Look into using multithreading and GPUs for these algorithms. They don't necessarily have advantage otherwise.
* There are some changes from the paper that I could make to the algorithms I have implemented, and they may make serious performance improvements.
    * Orthogonal basis/range finding
        * I could use specific types of random matrix that allows for faster matrix-matrix multiplies in one of the major bottlenecks.
        * There's an algorithm that automatically finds the approximate range of a matrix up to a fixed precision rather than a fixed rank.
            * It seems like we are more interested in fixing a rank a priori for computational efficiency or dimensionality reduction, so I am not sure this one is relevant.
            * Both fixed-rank and fixed-precision algorithms seem quite useful, so maybe I will just make both available.
    * SVD
        * There is an alternative algorithm that uses an interpolative decomposition to speed up SVD computation. However, this would require implementing an efficient interpolative decomposition subroutine, which seems non-trivial. I'll look around, but not sure if this one is worth the effort.
        * There are some single-pass algorithms, but the authors claim that they result in significant loss of precision, and should only be used if the entire matrix cannot be fit into memory.


###CCA
* Implement measure-transformed CCA.
    * Extend Zhuang Ma's gradient-based version to do the measure-transformed version.
        * For now, will still need to calculate the MT functions ahead of time.
* Do PCA dimensionality reduction of the data matrices before putting them into CCA. How does this interact with the data matrix whitening step? Equivalent?
    * Need to figure out how to do incremental PCA for the online setting. Does it maintain the PCA properties at every update?
        *Can use incremental PCA implementation from SciKitLearn.
* Try a few different stochastic gradient methods on CCA. Should take a look in Sebastian Bubeck's optimization survey to get a start. May want to try asynch stuff from Chris Re's asynch SGD review paper if I have time.
    * SVRG
    * RDA
    * AdaGrad
    * Natural gradient

###Bandits
* Implement the second meta-algorithm from Joulani's paper.
* Consider Kaplan-Meier estimates. Non-parametric version of maximum likelihood estimator of survival curve.

##Scaling
* If scaling becomes an issue, consider using SFrames.

##Documentation
* Figure out how to use Sphinx or Doxygen.
* Make notes in code about which papers are being implemented, and what we think might be a new method.

#Publication

##Delayed Rewards
* Start writing up the paper.

#Plots

##Delayed Rewards
* Cumulative reward over all plays, not just by action.

##CCA
* What kind of plots would be useful for CCA?
    * Maybe the kind from the gradient-based CCA paper where they compare the correlation captured by AppGrad to the correlation captured by naive CCA? That might require calling Matlab, though.
