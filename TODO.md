#Linear Algebra

##Sketching
* Set up the sketching code to use for the sketched second-order stuff.

#Data

##Missing
* Clean up the current missing data solution. Its a bit hacky and heavy-handed. I am not certain that there is a good single solution that integrates neatly into the data serving framework, but maybe we should have an entire subpackage that deals with missing data in different ways. That would be pretty awesome.

* Consider introducing monads into the data preparation/scrubbing pipeline.

##Servers
* Make a special data server for asynch-ish online optimization where its necessary to have a minimum batch size, e.g. because we need to do a rank _k_ update like in CCA.
    * Should have the option to either do shifting window updates or only update once we have a completely fresh batch. Is there a nice sliding scale between these two options?
    * Are there other problems that require rank _k_ updates and could benefit from such a scheme?
    * In general, consider interesting ways to maintain the data queue.

* Improve the bandit data server.

* Need to flip the data server/model relationship so that data servers can call on model subroutines. Then have bandits whose arms are models.

##Conversions
* Should maybe think about some pandas conversions for the data from the data servers. I seem to be writing these from scratch a bunch, especially for plotting. Is there a standard enough configuration so that I won't have to do that?

#Optimization

##Non-Stationary
* Try recomputing AdaGrad parameters on sliding window of data at each round. See if it keeps up with non-stationary simulated data.

* Think about data-dependent, adpative reweightings of gradients, but more time-sensitive, e.g. some sort of decaying exponential weight in addition to the AdaGrad parameters.

* In general, think about more sophisticated approaches to online optimization inspired by sliding-window regret.

##Step Size
* Generalize my code to allow for an arbitrary step-size scheduler.
    * What kind of arguments will they need to take?

* Fix the stepsizes for RDA and proximal gradient.

* Implement that one from Yann LeCun's student.

* Implement the probabilistic one.

* Implement Adam and AdaMax
