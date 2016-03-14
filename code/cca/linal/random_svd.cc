// Class header
#include "random_svd.h"

// Imports from C++
#include <iostream>

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
    int rank = k;

    if (k > max_rank)
    {
        std::cout << "WARNING: The value of k must not exceed the number of columns or rows of A. Setting k to maximum possible rank." << std::endl;
        rank = max_rank;
    }

    RandomOrthonormalBasis rob;
    MatrixXd Q = rob.GetRankKBasis(A, rank, q);

    std::cout << "Q cols: " << Q.cols() << " Q rows: " << Q.rows() << std::endl;

    Eigen::JacobiSVD<MatrixXd> svd(Q.transpose() * A, ComputeFullU | ComputeFullV);
    std::vector<MatrixXd> UsV = {Q * svd.matrixU(), svd.singularValues(), svd.matrixV()};

    return UsV;
}

} // namespace random
} // namespace linal
