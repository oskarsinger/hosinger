#Miscellaneous
* PUT CITATIONS IN YOUR CODE, e.g. simplex projection in optimization.utils

* Write some infrastructure to assist in the kind of exploratory data analysis described in the Gelman paper from 2004.
    * Is it possible to make this sufficiently general? Definitely should at least write a specific version for what I am working on.
    * This would be a great resume item for any of the Data Science for Social Good applications.

* Revamp CV
    * Clean up your optimization and CCA code, then put them on github
    * E4 project, both wavelets and online CCA, mainly wavelets since that's where the results are
    * Network interference project, complex optimization scheme synthesizing several recent publications and original ideas
    * Volunteering
    * Frisbee

#Network Interference

##Infrastructure
* Develop a Parameter class to take care of nasty indexing for stuff like FSVRG.

* Figure out how to nicely integrate data SERVERS in an RL context.

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
NOTE: This will have to wait until winter break or next semester probably.

* Try to find a way to propagate particle MCMC info efficiently across the network.

* Figure out exactly how to do particle MCMC and try to do some writing. Reading the Springer PDF a bit will probably help a lot.

###Federated
* Think carefully about when it is possible to do projected gradient in the context of federated, and how it can be accomplished.

* Play around with non-linear functions to replace A in order to deal with more difficult objective functions that are maybe not decomposable.

* Test the proposed advantages of FSVRG, e.g. drastically different numbers of parameters at each node, and different distributions at each node (although that doesn't seem to matter too much with complete independence across nodes as we currently have it).

* Try plugging in an AdaGrad transform at each node. May speed up convergence.

###RL Feedback
* So I def need to make a different parameter update depending on whether I apply treatment. How does that propagate through the link function, though? Gradient on log-likelihood of only parameters that involve the Rademachers? Do those depend on the Gaussian parameters too?

* Figure out how to integrate prox RL into my optimization library

#E4
* First show 'statistical picture' (CCA heat maps), then scatter plot, then individual example, then introduce likely causal relationship between accelerometer and heart rate

* Maybe switch all of my intermediate serialization to h5py. Will probably be cleaner and maybe more space-efficient.

##Experiments
* T-tests and p-values for spike in temperature vs reported symptoms

##Visualization
* Plot full time sequence instead of average over hours on each day

* Try completing the CCA over frequency time series like I did for partial reconstructions. For this, I need the sampling factor for the current subsampling rate, which will be the lowest one maintained for either of the wavelet matrices, I think? Whichever one corresponds to cca_dim. Do I need to be storing CCA dim so I can use it later, or can I find a way to pass it directly through?

* Don't show the element-wise multiplied singular vectors. Show them on different plots, side-by-side. Need to show all subjects on the same plot to compare across subjects.

* Once I get access to the server, run Al's plotting stuff and check his results against mine.

* Try averaging wavelet correlation (i.e. A matrix input for sparse CCA code) over days, but continue to do wavelet decomp over entire day at a time. Also, try doing the averaging over days on the small window again. Need to choose window size that results in good conditioning (i.e. 7 or fewer samples).
