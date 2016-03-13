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
* Test the accuracy of the randomized QR and SVD.
    * Test easy matrices like identity, diagonal, etc.
    * Write code to produce rank-k matrices.
    * Compare results of my code against results of Scikit-Learn's implementations.

####CCA
* Test CCA against Scikit-Learn's implementations.
    * Issue: the canonical basis returned by sklearn does not satisfy the quadratic norm orthonormality constraints of the CCA problem.
* Test CCA against Matlab's implementation.
* Get some real data from Yaya and run your code on it.
    * Issue: Yaya has not sent the data and doesn't respond to my emails about it.

####Bandits
* Test against various data scenarios:
    * Pre-set adversarial with varying levels of difficulty. 
        * Is there a 'smooth' function for tuning difficulty of an adversary?
    * Online adversarial with varying levels of difficulty.
        * Is there a 'smooth' function for tuning difficulty of an adversary?
    * Subroutines without BOLD, just one learner with cumulative rewards, where reward's causal action is visible to learner.
    * Warm start within BOLD.
    * Updating as we are waiting (censored observations).

###Efficiency
* Start doing stress tests on the scale that can be handled by the Python code.
    * This is especially relevant for CCA since computation for bandit algorithms is not super intense.
    * Do time trials to justify randomized linear algebra algorithms. Its getting faster (for low-rank matrices)!

* Do some more thorough benchmarking on the randomized linear algebra.
    * Probably need to find a decent benchmarking library.

##Code Structure

###Bandit Framework

###CCA
* Should I try to put any framework-y type structure on the CCA stuff?

###C++/Python Interaction

##Algorithms

###Randomized Linear Algebra
* There are some changes from the paper that I could make to the algorithms I have implemented, and they may make serious performance improvements.
    * Orthogonal basis/range finding
        * I could use a specific type of random matrix that allows for faster matrix-matrix multiplies in one of the major bottlenecks.
            * This structured matrix seems quite tricky to produce, so I am questioning the net worth of that performance enhancement.
            * They give the structured matrix for the complex case, but don't explain how to deal with the real case. I'll look around for other resources, though. There may be something in their citation list.
        * There's an algorithm that automatically finds the approximate range of a matrix up to a fixed precision rather than a fixed rank.
            * It seems like we are more interested in fixing a rank a priori for computational efficiency or dimensionality reduction, so I am not sure this one is relevant.
            * Both fixed-rank and fixed-precision algorithms seem quite useful, so maybe I will just make both available.
    * SVD
        * There is an alternative algorithm that uses an interpolative decomposition to speed up SVD computation. However, this would require implementing an efficient interpolative decomposition subroutine, which seems non-trivial. I'll look around, but not sure if this one is worth the effort.
        * There are some single-pass algorithms, but the authors claim that they result in significant loss of precision, and should only be used if the entire matrix cannot be fit into memory.

###CCA
* Implement measure-transformed CCA.
    * Try the classical approach to CCA calculation.
    * Extend Zhuang Ma's gradient-based version to do the measure-transformed version.
        * For now, will still need to calculate the MT functions ahead of time.
* I wonder if the proof for convergence of the gradient-based CCA will be different for the measure-transformed version?
* Farm out some of the more expensive subroutines to C++.

###Bandits
* Implement the second meta-algorithm from Joulani's paper.
* Consider Kaplan-Meier estimates. Non-parametric version of maximum likelihood estimator of survival curve.

##Scaling
* If scaling becomes an issue, consider using SFrames.

##Documentation
* Figure out how to use Sphinx or Doxygen.

#Publication

##Delayed Rewards
* Start writing up the paper.

#Plots

##Delayed Rewards
* Arm counts vs time
    * either by n-length period or by individual time step (cumulative percentage of arm usage at each time step or period).
* Reward vs time.
    * Percentage of optimal. Should be increasing.
* Or regret vs time.
    * Difference from optimal regret. Should be decreasing.
