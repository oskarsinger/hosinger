#Linear Algebra

##Decompositions

###QR
* Need to make sure that my inner product space QR decomposition will be numerically stable.
* Need to try to take advantage of block diagonal structure.
    * Should start by feeding in some compact representation of block locations and calling current function as subroutine on each block.

#Data
* Choose either servers or loaders to give option to average over certain coordinates.

##Servers
* Make a special data server for asynch-ish stochastic gradient where its necessary to have a minimum batch size, e.g. because we need to do a rank _k_ update like in CCA.
    * Should have the option to either do shifting window updates or only update once we have a completely fresh batch. Is there a nice sliding scale between these two options?
    * Are there other problems that require rank _k_ updates and could benefit from such a scheme?

##Loaders
