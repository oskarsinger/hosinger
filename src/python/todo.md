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

###Online EM
* Need to figure out how to fit the E step into federated optimization. Probably can just separately implement federated optimization, figure out the search direction of the E step, then plug into federated as if it were a gradient.

###Federated
* Carefully consider how to define A and S in my implementation of federated optimization. Should I make separate A and S for $\mu$'s and $p$'s or the same one? Its going to double my number of parameters, but is that really so bad?

* Since the coordinates may not be as independent as assumed in the McMahan paper, it may be a good idea to use that sketching for online second-order methods paper. Yay! Will have to run experiments with both.

* Gotta use entropic mirror descent on the ps. Can that be working into the federated optimization framework? Most likely yeah. Does entropic just involve projecting onto [0,1]? Of course, maybe EM takes that into account?

#E4
* First show 'statistical picture' (CCA heat maps), then scatter plot, then individual example, then introduce likely causal relationship between accelerometer and heart rate

##Experiments
* T-tests and p-values for spike in temperature vs reported symptoms

* Transpose the frequency component matrices and apply sparse CCA to find events.

##Visualization
* Plot full time sequence instead of average over hours on each day
