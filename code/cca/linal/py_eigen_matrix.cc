#include "py_eigen_matrix.h"

namespace linal {
namespace python {

PyEigenMatrixXd::PyEigenMatrixXd(const std::vector<std::vector<double> > &initial)
{
    this->matrix = Eigen::MatrixXd::Zero(rows, cols);

    auto outer_first = initial.begin();

    for (auto outer = outer_first; outer != inital.end(); ++outer)
    {
        auto inner_first = outer->begin();

        for (auto inner = inner_first; inner != outer->end(); ++inner)
        {
            this->matrix(outer - outer_first, inner - inner_first) = *inner;
        }
    }
}

PyEigenMatrixXd::PyEigenMatrixXd(const Eigen::MatrixXd &initial)
{
    this->matrix = initial;
}

std::vector<std::vector<double>* >* to_vector()
{
    int rows = this->matrix.rows();
    int cols = this->matrix.cols();

    std::vector<std::vector<double> *> *vec = new std::vector<std::vector<double > >(rows, new std::vector<double>(cols, 0));

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols;, ++j)
        {
            (vec->at(i))->at(j) = this->matrix(i,j);
        }
    }

    return vec;
}

} // namespace python
} // namespace linal
