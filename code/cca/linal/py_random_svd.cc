// Class header
#include "py_random_svd.h"

// Imports from C++
#include <iostream>

namespace linal {
namespace python {

using namespace Eigen;

const random::RandomSvd PyRandomSvd::rSvd = random::RandomSvd();

PyRandomSvd::PyRandomSvd(const std::vector<std::vector<double> > &initial)
    : PyEigenMatrixXd(initial)
{}

void PyRandomSvd::GetRandomSvd()
{
    fill_svd(this->rSvd.GetRandomSvd(this->matrix));
}

void PyRandomSvd::GetRandomSvd(const int k)
{
    fill_svd(this->rSvd.GetRandomSvd(this->matrix, k));
}

void PyRandomSvd::GetRandomSvd(const int k, const int q)
{
    fill_svd(this->rSvd.GetRandomSvd(this->matrix, k, q));
}

void PyRandomSvd::fill_svd(const std::vector<MatrixXd> &matrices)
{
    this->U = PyEigenMatrixXd(matrices[0]).to_vector();
    this->s = PyEigenMatrixXd(matrices[1]).to_vector();
    this->V = PyEigenMatrixXd(matrices[2]).to_vector();
}

} // namespace python
} // namespace linal
