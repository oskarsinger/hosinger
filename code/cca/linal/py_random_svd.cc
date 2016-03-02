// Class header
#include "py_random_svd.h"

// Imports from C++
#include <iostream>

namespace linal {
namespace python {

using namespace Eigen;

const random::RandomSvd PyRandomSvd::rSvd = random::RandomSvd();

std::vector<PyEigenMatrixXd> PyRandomSvd::GetRandomSvd(const PyEigenMatrixXd &A) const
{
    MatrixXd matrix = A.matrix;

    return WrapVector(this->rSvd.GetRandomSvd(matrix));
}

std::vector<PyEigenMatrixXd> PyRandomSvd::GetRandomSvd(const PyEigenMatrixXd &A, const int k) const
{
    MatrixXd matrix = A.matrix;

    return WrapVector(this->rSvd.GetRandomSvd(matrix, k));
}

std::vector<PyEigenMatrixXd> PyRandomSvd::GetRandomSvd(const PyEigenMatrixXd &A, const int k, const int q) const
{
    MatrixXd matrix = A.matrix;

    return WrapVector(this->rSvd.GetRandomSvd(matrix, k, q));
}

std::vector<PyEigenMatrixXd> PyRandomSvd::WrapVector(const std::vector<MatrixXd> &matrices) const
{
    std::vector<PyEigenMatrixXd> wrapped = std::vector<PyEigenMatrixXd>();

    for (auto itr = matrices.begin(); itr != matrices.end(); ++itr)
    {
        wrapped.push_back(PyEigenMatrixXd(*itr));
    }

    return wrapped;
}

} // namespace python
} // namespace linal
