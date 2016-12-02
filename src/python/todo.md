#Miscellaneous
* Write some infrastructure to assist in the kind of exploratory data analysis described in the Gelman paper from 2004.
    * Is it possible to make this sufficiently general? Definitely should at least write a specific version for what I am working on.
    * This would be a great resume item for any of the Data Science for Social Good applications.

* Revamp CV; maybe put some code on github (which code? I don't currently have anything worth putting up; should change that).

#Network Interference

##Infrastructure
* Data loader shortcuts, specifically randomly generated ones. Should I try to fit this into the same bandit arm servers I was using earlier?

* Click tool for running the runner

##Learning
* Should I start with a global model at each node? Yes. So I need to tell each learner about network structure at the beginning.

* Eventually, I need to be concerned about subsampling nodes in the network for using in the weighted proximal updates. Need to come up with a sampling scheme, and I should ask Walter and Brandon about this. Should also take a look at federated optimization paper.
    
##Network
* How should I simulate limited availability of nodes? The learners should be communicating with each other internally rather than having availability fed into them. So they should loop through the other nodes checking for availability

##Visualization
* Need to make visualization to assess convergence of the algorithm on the local parameters. Should I also be assessing convergence on global model, or does that matter? Make that a later step? Maybe I should show the max error over all nodes for each coordinate.

##Optimization
* Consider replacing the step of distributing the global gradient approximation with some other form of information propagation across the network. Maybe try using the multiprocessing library's tools for sharing state across threads/processes? Save this guy for later. For now just collect the process results.

* Write up model to represent linear regression. Include expression for gradient, number of parameters, etc. How will I get number of parameters without seeing the data?

* Implementation steps for online FSVRG:
    1. Original SVRG
    2. Infrastructure for testing distributed algs
        * Needs to allow for randomly unavailable nodes. Would be nice if I didn't have to simulate that inside of SVRG subroutine. Maybe write a wrapper that simulates unavailability.
    3. Distributed SVRG, AIDE, DANE, etc.
    4. Original FSVRG, hopefully fits in testing framework from item 2.
    5. More online-ish FSVRG

###Particle MCMC
NOTE: This will have to wait until winter break or next semester probably.

* Try to find a way to propagate particle MCMC info efficiently across the network.

* Figure out exactly how to do particle MCMC and try to do some writing. Reading the Springer PDF a bit will probably help a lot.

###Federated
* Need to figure out best way to share information from other GMMs with federated optimizer. Now that I understand this part, will hopefully be much easier to define A and S.

###Bandit Feedback

#E4
* First show 'statistical picture' (CCA heat maps), then scatter plot, then individual example, then introduce likely causal relationship between accelerometer and heart rate

##Experiments
* T-tests and p-values for spike in temperature vs reported symptoms

##Visualization
* Plot full time sequence instead of average over hours on each day

* Try completing the CCA over frequency time series like I did for partial reconstructions

* Don't show the element-wise multiplied singular vectors. How to show it then? Show both left and right vectors on the same plot or different ones? Side by side?

* Compare my partial reconstructions to Al's

* Try averaging wavelet correlation (i.e. A matrix for sparse library) over days, but continue to do wavelet decomp over entire day at a time. Also, try doing the averaging over days on the small window again. Need to choose window size that results in good conditioning (i.e. 7 or fewer samples).
