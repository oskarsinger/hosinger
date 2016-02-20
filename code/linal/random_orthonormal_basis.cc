// Class header
#include "random_rank_k_basis.h"

// Imports from C++
#include <iostream>
#include <time.h>

// Imports from external projects
#include <Eigen/QR>

// Imports from this project
#include "random_matrix_factory.h"

namespace linal {
namespace random {

Eigen::MatrixXd RandomOrthonormalBasis::GetEpsilonBasis(Eigen::MatrixXd A, const double epsilon)
{
    std::cout << "WARNING: this method is not implemented and will return a zero matrix." << std::endl;
    return Eigen::MatrixXd::Zero(A.rows(), A.cols());
}

Eigen::MatrixXd RandomOrthonormalBasis::GetFullRankBasis(Eigen::MatrixXd A)
{
    int m = A.rows();
    int n = A.cols();
    int max_rank = std::min(m,n);

    return GetRankKBasis(A, max_rank);
}

Eigen::MatrixXd RandomOrthonormalBasis::GetRankKBasis(Eigen::MatrixXd A, const int k)
{
    return GetRankKBasis(A, k, 1);
}

Eigen::MatrixXd RandomOrthonormalBasis::GetRankKBasis(Eigen::MatrixXd A, const int k, const int q)
{
    int m = A.rows(); 
    int n = A.cols();
    int max_rank = std::min(m,n);

    if (k > max_rank)
    {
        std::cout << "WARNING: The value of k must not exceed the number of columns or rows of A." << std::endl;
    }

    time_t before, after;

    time(&before);
    RandomMatrixFactory fac();
    Eigen::MatrixXd Y = A * fac::GetNormalMatrix(n, k);
    time(&after);
    std::cout << "Acquired random matrix" << difftime(after, before) << std::endl;

    time(&before);
    Eigen::MatrixXd Q = Eigen::HouseholderQR<Eigen::MatrixXd>(Y).householderQ();
    time(&after);
    std::cout << "Got first QR" << difftime(after, before) << std::endl;

    time(&before);
    for (int i = 0; i < q; i++)
    {
        Y = A.transpose() * Q;
        Q = Eigen::HouseholderQR<Eigen::MatrixXd>(Y).householderQ();
        Y = A * Q;
        Q = Eigen::HouseholderQR<Eigen::MatrixXd>(Y).householderQ();
    }
    time(&after);
    std::cout << "Completed power iteration" << difftime(after, before) << std::endl;

    return Q;
}

} // namespace random
} // namespace linal
