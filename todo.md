#Engineering

##Tests

###Correctness
* Test the validity of the implementations.
    * Write up some unit tests in nosetest for
        * CCA
        * SVD
        * QR
        * linal utility functions
        * exp3
        * ucb1

###Accuracy
* Write up a somewhat general framework for running experiments.
    * Would be nice if it automatically printed charts/graphs/tables.

* Test the accuracy of the randomized QR and SVD.
    * Test easy matrices like identity, diagonal, etc.
    * Write code to produce rank-k matrices.
    * Compare results of my code against results of Scikit-Learn's implementations.

* Test CCA against Scikit-Learn's implementations.

###Efficiency
* Start doing stress tests on the scale that can be handled by the Python code.
    * This is especially relevant for CCA since computation for bandit algorithms is not super intense.
    * Do time trials to justify randomized linear algebra algorithms.

##Documentation
* Figure out how to use Sphinx or Doxygen.
