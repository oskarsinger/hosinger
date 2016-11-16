#Network Interference
* Split APE file of Trio of Doom album into individual tracks

##Infrastructure
* Data loader shortcuts, specifically randomly generated ones. Should I try to fit this into the same bandit arm servers I was using earlier?

* Click tool for running the runner

##Learning
* Need to come up with incremental alternating updates to each node's parameters.

* Should I start with a global model at each node? Yes. So I need to tell each learner about network structure at the beginning.

* Eventually, I need to be concerned about subsampling nodes in the network for using in the weighted proximal updates. Need to come up with a sampling scheme, and I should ask Walter and Brandon about this. Should also take a look at federated optimization paper.
    
##Network
* How should I simulate limited availability of nodes? The learners should be communicating with each other internally rather than having availability fed into them. So they should loop through the other nodes checking for availability

##Visualization
* Need to make visualization to assess convergence of the algorithm on the local parameters. Should I also be assessing convergence on global model, or does that matter? Make that a later step? Maybe I should show the max error over all nodes for each coordinate.

##Optimization
* I should just write out the actual prox operator with the dual averaging. Otherwise, I am never going to figure this out.

###Particle MCMC
* Try to find a way to propagate particle MCMC info efficiently across the network. Could just use some basic communication algs since the local parameters themselves don't actually need the information propagation.

* Figure out exactly how to do particle MCMC and try to do some writing.

###Online EM
* Just write out the distribution of $Y_v^{(t)}$ in terms of the parameters known from the burn-in time so you can account for the baseline effects when doing the EM updates. Also make sure to account for the exposure effects by either adding or subtracting according to $j$ when finding the weighted empirical moments.

* Need to figure out how to fit the E step into federated optimization. Probably can just separately implement federated optimization, figure out the search direction of the E step, then plug into federated as if it were a gradient.

* Now that I think about it, the M step may also need to involve federated optimization, although the M step sort of gets the federated part via the E step because its working with the FDO-reweighted estimates of the $p$s.

###Federated
* Need to figure out best way to share information from other GMMs with federated optimizer. Now that I understand this part, will hopefully be much easier to define A and S.

###Bandit Feedback
* The bandit feedback comes from the fact that we won't be making treatment on our subjects at each round. However, even without treatment-related feedback, we are still getting feedback from the deterministic exposure effects, but I guess this update just applies to the Rademacher variables. Wait, ew, how do I just update the Rademachers?

#E4
* First show 'statistical picture' (CCA heat maps), then scatter plot, then individual example, then introduce likely causal relationship between accelerometer and heart rate

##Experiments
* T-tests and p-values for spike in temperature vs reported symptoms

* Fix partial reconstruction. Not really sure how to debug it. I think I'll need to have a better understanding of the DTCWT before I can intelligently approach depbugging the partial reconstruction.

##Visualization
* Plot full time sequence instead of average over hours on each day
