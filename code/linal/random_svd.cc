// Class header
#include "random_svd.h"

// Imports from C++
#include <iostream>
#include <time.h>

// Imports from external projects
#include <Eigen/SVD>

// Imports from this project
#include "random_orthonormal_basis.h"

namespace linal {
namespace random {

std::vector<MatrixXd> RandomSVD::get_random_svd(MatrixXd A)
{
    int m = A.rows();
    int n = A.cols();
    int max_rank = std::min(m, n);

    return get_random_svd(A, max_rank);
}

std::vector<MatrixXd> RandomSVD::get_random_svd(MatrixXd A, const int k)
{
    return get_random_svd(A, k, 1);
}

std::vector<MatrixXd> RandomSVD::get_random_svd(MatrixXd A, const int k, const int q)
{
    int n = A.rows();
    int m = A.cols();
    int max_rank = std::min(n,m);

    if (k > max_rank)
    {
        std::cout << "WARNING: The value of k must not exceed the number of columns or rows of A." << std::endl;
    }

    RandomOrthonormalBasis rob();
    MatrixXd Q = rob::get_orthonormal_basis(A, k, q);

    time_t before, after;

    time(&before);
    Eigen::JacobiSVD<MatrixXd> svd(Q.transpose() * A, ComputeFullU | ComputeFullV);
    time(&after);
    std::cout << "Completed SVD" << difftime(after, before) << std::endl;

    MatrixXd U = Q * svd.matrixU();
    std::vector<MatrixXd> UsV = {U, svd.singularValues(), svd.matrixV()};

    return UsV;
}

} // namespace random
} // namespace linal
