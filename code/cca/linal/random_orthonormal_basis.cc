// Class header
#include "random_orthonormal_basis.h"

// Imports from C++
#include <iostream>

// Imports from external projects
#include <Eigen/QR>

// Imports from this project
#include "random_matrix_factory.h"

namespace linal {
namespace random {

using namespace Eigen;

MatrixXd RandomOrthonormalBasis::GetEpsilonBasis(const MatrixXd &A, const double epsilon)
{
    std::cout << "WARNING: this method is not implemented and will return a zero matrix." << std::endl;
    return MatrixXd::Zero(A.rows(), A.cols());
}

MatrixXd RandomOrthonormalBasis::GetFullRankBasis(const MatrixXd &A)
{
    int m = A.rows();
    int n = A.cols();
    int max_rank = std::min(m,n);

    return GetRankKBasis(A, max_rank);
}

MatrixXd RandomOrthonormalBasis::GetRankKBasis(const MatrixXd &A, const int k)
{
    return GetRankKBasis(A, k, 1);
}

MatrixXd RandomOrthonormalBasis::GetRankKBasis(const MatrixXd &A, const int k, const int q)
{
    int m = A.rows(); 
    int n = A.cols();
    int max_rank = std::min(m,n);

    if (k > max_rank)
    {
        std::cout << "WARNING: The value of k must not exceed the number of columns or rows of A." << std::endl;
    }

    RandomMatrixFactory rmf;
    MatrixXd Q = Eigen::HouseholderQR<MatrixXd>(A * rmf.GetNormalMatrix(n,k)).householderQ();

    for (int i = 0; i < q; i++)
    {
        Q = Eigen::HouseholderQR<MatrixXd>(A.transpose() * Q).householderQ();
        Q = Eigen::HouseholderQR<MatrixXd>(A * Q).householderQ();
    }

    std::cout << "Q cols: " << Q.cols() << " Q rows: " << Q.rows() << std::endl;

    return Q;
}

} // namespace random
} // namespace linal
