// Class header
#include "random_matrix_factory.h"

// Imports from C++
#include <iostream>
#include <random>

namespace linal {
namespace random {

Eigen::MatrixXd RandomMatrixFactory::GetRankKMatrix(const int m, const int n, const int k)
{
    int max_rank = std::min(m,n);

    if (k > max_rank)
    {
        std::cout << "WARNING: The value of k must not exceed the number of columns or rows." << std::endl;
    }

    MatrixXd A = MatrixXd::Zero(m,n);

    for (int i = 0; i < k; ++i)
    {
        A += MatrixXd::Random(m,1) * MatrixXd::Random(1,n);
    }

    return A;
}

Eigen::MatrixXd RandomMatrixFactory::GetNormalMatrix(const int m, const int n)
{
    //Random number generation stuff
    std::random_device rd;
    std::mt19937 e2(rd());
    std::normal_distribution<> dist(0,1);

    //Random matrix to be initialized
    MatrixXd A(m, n);

    //Populate entries of random matrix
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            A(i, j) = dist(e2);
        }
    }

    return A;
}

} // namespace random
} // namespace linal
