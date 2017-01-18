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

##Experiments
* T-tests and p-values for spike in temperature vs reported symptoms

##Visualization
* Show correlation of the partial reconstructions maybe?

* Currently just putting the same frequency, same view, different subjects on same plot. Could my analysis efforts benefit from different grouping?

* How do I show that the multiview is more informative? Should I start computing plots for more than just HR and ACC?

* Do vertically stacked plots for CCA components instead of plots of vertically stacked components

* Implement view-hour-pairwise correlation

* How can I abstract away the awkward nested iteration for the correlation analysis? May not be possible.

* In general, work on speeding up these tools. They are pretty slow right now, and I could probably take more advantage of numpy's performance improvements by doing more reshaping and less looping, etc.
    * Should I just server up the full, concatenate dataset, without period-wise separation, then use numpy operations to do all the averaging and such? Probalby can clean-up/speed-up that way. Worry about that later.
