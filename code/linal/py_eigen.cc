#include "py_eigen.h"

namespace linal {
namespace python {

PyEigen::PyEigen(int rows, int cols)
{
    matrix = Eigen::MatrixXd::Zero(rows, cols);
}

double PyEigen::Get(int row, int col)
{
    return matrix(row, col);
}

void PyEigen::Set(int row, int col, double val)
{
    matrix(row, col) = val;
}

int PyEigen::Rows()
{
    return matrix.rows();
}

int PyEigen::Cols()
{
    return matrix.cols();
}

} // namespace python
} // namespace linal
