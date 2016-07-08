#Linear Algebra

##Decompositions

###QR
* Need to make sure that my inner product space QR decomposition will be numerically stable.

* Need to try to take advantage of block diagonal structure.
    * Should start by feeding in some compact representation of block locations and calling current function as subroutine on each block.

#Data
* Choose either servers or loaders to give option to average over certain coordinates. Probably servers so I don't have to implement the same thing multiple times.

##Servers
* Make a special data server for asynch-ish stochastic gradient where its necessary to have a minimum batch size, e.g. because we need to do a rank _k_ update like in CCA.
    * Should have the option to either do shifting window updates or only update once we have a completely fresh batch. Is there a nice sliding scale between these two options?
    * Are there other problems that require rank _k_ updates and could benefit from such a scheme?
    * In general, consider interesting ways to maintain the data queue.

* Probably need to rework the Gaussian data server.
    * Can probably merge the original and shifting-mean implementations if I can figure out how to reconcile the minibatch subsampling and pure streaming approaches.

* Make a data server to serve up periodic data.

##Step Size
* Generalize my code to allow for an arbitrary step-size scheduler.
    * What kind of arguments will they need to take?

* Implement that one from Yann LeCun's student.

* Implement the probabilistic one.

* Implement the one that was mentioned in the optimization-online update.
