#include "py_eigen_matrix.h"

namespace linal {
namespace python {

PyEigenMatrixXd::PyEigenMatrixXd(const std::vector<std::vector<double> > &initial)
{
    int rows = initial.size();
    int cols = initial[0].size();

    this->matrix = Eigen::MatrixXd::Zero(rows, cols);

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            this->matrix(i,j) = initial[i][j];
        }
    }
}

PyEigenMatrixXd::PyEigenMatrixXd(const Eigen::MatrixXd &initial)
{
    this->matrix = initial;
}

std::vector<std::vector<double> > PyEigenMatrixXd::to_vector()
{
    int rows = this->matrix.rows();
    int cols = this->matrix.cols();

    std::vector<std::vector<double> > vec = std::vector<std::vector<double > >(rows, std::vector<double>(cols, 0));

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            vec[i][j] = this->matrix(i,j);
        }
    }

    return vec;
}

} // namespace python
} // namespace linal
