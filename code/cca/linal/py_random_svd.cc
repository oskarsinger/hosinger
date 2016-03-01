// Class header
#include "py_random_svd.h"

// Imports from C++
#include <iostream>

namespace linal {
namespace python {

using namespace Eigen;

std::vector<PyEigenMatrixXd> PyRandomSvd::GetRandomSvd(PyEigenMatrixXd A)
{
    MatrixXd matrix = A.matrix;

    return WrapVector(rSvd.GetRandomSvd(matrix));
}

std::vector<PyEigenMatrixXd> PyRandomSvd::GetRandomSvd(PyEigenMatrixXd A, const int k);
{
    MatrixXd matrix = A.matrix;

    return WrapVector(rSvd.GetRandomSvd(matrix, k));
}

std::vector<PyEigenMatrixXd> PyRandomSvd::GetRandomSvd(PyEigenMatrixXd A, const int k, const int q);
{
    MatrixXd matrix = A.matrix;

    return WrapVector(rSvd.GetRandomSvd(matrix, k, q));
}

std::vector<PyEigenMatrixXd> WrapVector(std::vector<MatrixXd> matrices)
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
