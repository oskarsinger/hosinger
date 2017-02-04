#Miscellaneous
* PUT CITATIONS IN YOUR CODE, e.g. simplex projection in optimization.utils, federated, SVRG, AdaGrad, etc, etc

* Write some infrastructure to assist in the kind of exploratory data analysis described in the Gelman paper from 2004.
    * Is it possible to make this sufficiently general? Definitely should at least write a specific version for what I am working on.
    * This would be a great resume item for any of the Data Science for Social Good applications.

* Revamp CV
    * Clean up your optimization and CCA code, then put them on github
    * E4 project, both wavelets and online CCA, mainly wavelets since that's where the results are
    * Network interference project, complex optimization scheme synthesizing several recent publications and original ideas
    * Volunteering
    * Frisbee

* Create hdf5 serialization helpers for standard datatypes if they don't already exist.

#Network Interference

##Infrastructure
* Develop a Parameter class to take care of nasty indexing for stuff like FSVRG. That actually may not be necessary until we have more complex parameter sharing and feature generation stuff.

##Learning
* Eventually, I need to be concerned about subsampling nodes in the network for using in the weighted proximal updates. Need to come up with a sampling scheme, and I should ask Walter and Brandon about this. Should also take a look at federated optimization paper. Also, turns out Brendan McMahan and co have a paper about this that could be helpful, although I think they sample at each round, and I'd want to have probably a subset at the beginning. That may be more related to the model than the estimation. Let's see where they take the SBM stuff.
    
##Network

##Visualization
* Need to make visualization to assess convergence of the algorithm on the local parameters. Should I also be assessing convergence on global model, or does that matter? Make that a later step? Maybe I should show the max error over all nodes for each coordinate.

* Need to compare convergence of nodes w.r.t. to frequency of treatment.

##Optimization
* Replace the step of distributing the global gradient approximation with some other form of information propagation across the network. Maybe try using the multiprocessing library's tools for sharing state across threads/processes? Save this guy for later. For now just collect the process results.

* Look at the plots produced in the AIDE, SVRG, and Federated papers and reproduce them for your own experiments.

* Implementation steps for online FSVRG:
    2. Infrastructure for testing distributed algs
        * Needs to allow for randomly unavailable nodes. Would be nice if I didn't have to simulate that inside of SVRG subroutine. Maybe write a wrapper that simulates unavailability.

###DIAG
* Implement static, single-threaded DIAG

* Derive and implement 'federated' DIAG and RLFDIAG

###Quasinewton Servers
* Implement full AdaGrad and Adam, diagonal Adam

* Implement the paper on low-rank estimates of QN matrices

###Particle MCMC
* Browse the Springer PDF!

* Go to the RM-HMC reading group, and read the materials! Think about how it fits into the framework you are already using.

###Federated
* Think carefully about when it is possible to do projected gradient in the context of federated, and how it can be accomplished. Certainly, you need to be doing it before using your most recent model to make a prediction. Maybe elsewhere to keep things on track.

* Play around with non-linear functions to replace A in order to deal with more difficult objective functions that are maybe not decomposable.

* Test the proposed advantages of FSVRG, e.g. drastically different numbers of parameters at each node, and different distributions at each node (although that doesn't seem to matter too much with complete independence across nodes as we currently have it).

* Try plugging in an AdaGrad transform at each node. May speed up convergence.

###RL Feedback
* So I def need to make a different parameter update depending on whether I apply treatment. How does that propagate through the link function, though? Gradient on log-likelihood of only parameters that involve the Rademachers? Do those depend on the Gaussian parameters too?

* Figure out how to integrate prox RL into my optimization library, or maybe just whether it should be added. Probably it should just be something inside an RL module that calls my prox optimization stuff.

#E4
* First show 'statistical picture' (CCA heat maps), then scatter plot, then individual example, then introduce likely causal relationship between accelerometer and heart rate

##Experiments
* T-tests and p-values for spike in temperature vs reported symptoms

##Visualization
* Look at error of partial reconstructions rather than the reconstructions themselves.

* Show correlation of the partial reconstructions maybe?

* Produce some autocorrelation plots for the E4 data.

* Fix Al's Matlab script to work with CM data.

* Fix the NaN-valued wavelets in the CCA code.

* Maybe plug standard deviation into the wavelet decomposition

* Show similar R squared plots to Yaya's by regressing on E4 data, wavelet coefficients, CCA-filtered data, canonical correlation
    * It will help to build all off the above post-processing into data servers; consider making an mvc submodule for the data.servers.masks submodule.
