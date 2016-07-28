#Linear Algebra

##Decompositions

###QR
* Need to make sure that my inner product space QR decomposition will be numerically stable.

* Need to try to take advantage of block diagonal structure.
    * Should start by feeding in some compact representation of block locations and calling current function as subroutine on each block.

#Data

##Missing
* Clean up the current missing data solution. Its a bit hacky and heavy-handed. I am not certain that there is a good single solution that integrates neatly into the data serving framework, but maybe we should have an entire subpackage that deals with missing data in different ways. That would be pretty awesome.

* Consider introducing monads into the data preparation/scrubbing pipeline.

##Loaders
* Synthetic periodic data loader.

##Servers
* Make a special data server for asynch-ish stochastic gradient where its necessary to have a minimum batch size, e.g. because we need to do a rank _k_ update like in CCA.
    * Should have the option to either do shifting window updates or only update once we have a completely fresh batch. Is there a nice sliding scale between these two options?
    * Are there other problems that require rank _k_ updates and could benefit from such a scheme?
    * In general, consider interesting ways to maintain the data queue.

* Probably need to rework the Gaussian data server.
    * Can probably merge the original and shifting-mean implementations if I can figure out how to reconcile the minibatch subsampling and pure streaming approaches.

* Improve the bandit data server.

##Conversions
* Should maybe think about some pandas conversions for the data from the data servers. I seem to be writing these from scratch a bunch, especially for plotting. Is there a standard enough configuration so that I won't have to do that?

#Models

##Data Consumption
* It would be nice if there were a mechanism by which to externally invoke a specific number of model updates on a specific server instead of having to build the servers around the constraint that once they go into the model, we lose control of them. This would also fit our impending need to work within a filtering scenario much better.
    * One issue I see with this is that we would have to disentagle some of the interation logic from the models themselves, which seems undesirable and painful. Should think carefully about whether its worth the effort or whether this detriment actually exists.

#Optimization

##Distributed/Asynch
* (Online?) ADMM with Newton Sketch. Reread the 'Dual Averaging and Proximal Gradient Descent for Online ADMM' paper for implementation help. Need to read Newton Sketch too.

##Matrix-Valued FTPRL
* Should I be applying operations like accumulation of AdaGrad parameters and shrinkage-and-thresholding to just eigen/singular values or to the full parameter matrix? I think it depends on the situation. Should probably offer both options.

##Non-Stationary
* Try recomputing AdaGrad parameters on sliding window of data at each round. See if it keeps up with non-stationary simulated data.

* Think about data-dependent, adpative reweightings of gradients, but more time-sensitive, e.g. some sort of decaying exponential weight in addition to the AdaGrad parameters.

* In general, think about more sophisticated approaches to online optimization inspired by sliding-window regret.

##Step Size
* Generalize my code to allow for an arbitrary step-size scheduler.
    * What kind of arguments will they need to take?

* Implement that one from Yann LeCun's student.

* Implement the probabilistic one.
