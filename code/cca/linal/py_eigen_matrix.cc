#include "py_eigen_matrix.h"

namespace linal {
namespace python {

PyEigenMatrixXd::PyEigenMatrixXd(int rows, int cols)
{
    matrix = Eigen::MatrixXd::Zero(rows, cols);
}

PyEigenMatrixXd::PyEigenMatrixXd(Eigen::MatrixXd initial)
{
    matrix = initial;
}

double PyEigenMatrixXd::Get(int row, int col)
{
    return matrix(row, col);
}

void PyEigenMatrixXd::Set(int row, int col, double val)
{
    matrix(row, col) = val;
}

int PyEigenMatrixXd::Rows()
{
    return matrix.rows();
}

int PyEigenMatrixXd::Cols()
{
    return matrix.cols();
}

} // namespace python
} // namespace linal
