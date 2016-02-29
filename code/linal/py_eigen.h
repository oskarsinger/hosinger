#ifndef PY_EIGEN_H_
#define PY_EIGEN_H_

// Imports from other projects
#include <Eigen/Dense>

namespace linal {
namespace python {

class PyEigen
{
 public:
  PyEigen(int rows, int cols);
  ~PyEigen() {}
  double Get(int row, int col);
  void Set(int row, int col, double val);
  int Rows();
  int Cols();

 private:
  Eigen::MatrixXd matrix;
};

} // namespace python
} // namespace linal

#endif // EIGEN_WRAPPER_H_
