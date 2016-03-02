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

using namespace Eigen;

std::vector<MatrixXd> RandomSvd::GetRandomSvd(const MatrixXd &A) const
{
    int m = A.rows();
    int n = A.cols();
    int max_rank = std::min(m, n);

    return GetRandomSvd(A, max_rank);
}

std::vector<MatrixXd> RandomSvd::GetRandomSvd(const MatrixXd &A, const int k) const
{
    return GetRandomSvd(A, k, 1);
}

std::vector<MatrixXd> RandomSvd::GetRandomSvd(const MatrixXd &A, const int k, const int q) const
{
    int n = A.rows();
    int m = A.cols();
    int max_rank = std::min(n,m);

    if (k > max_rank)
    {
        std::cout << "WARNING: The value of k must not exceed the number of columns or rows of A." << std::endl;
    }

    RandomOrthonormalBasis rob;
    MatrixXd Q = rob.GetRankKBasis(A, k, q);

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
