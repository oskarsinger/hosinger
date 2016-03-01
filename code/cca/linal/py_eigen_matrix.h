#ifndef PY_EIGEN_H_
#define PY_EIGEN_H_

// Imports from other projects
#include <Eigen/Dense>

namespace linal {
namespace python {

class PyEigenMatrixXd
{
 public:
  // Constructor and destructor
  PyEigenMatrixXd(int rows, int cols);
  PyEigenMatrixXd(Eigen::MatrixXd initial);
  ~PyEigenMatrixXd() {}

  // Methods
  double Get(int row, int col);
  void Set(int row, int col, double val);
  int Rows();
  int Cols();

  // Data members
  Eigen::MatrixXd matrix;
};

} // namespace python
} // namespace linal

#endif // EIGEN_WRAPPER_H_
