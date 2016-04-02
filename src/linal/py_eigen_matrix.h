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
  PyEigenMatrixXd(const std::vector<std::vector<double> > &initial);
  PyEigenMatrixXd(const Eigen::MatrixXd &initial);
  ~PyEigenMatrixXd() {}

  std::vector<std::vector<double> > to_vector();

  // Data members
  Eigen::MatrixXd matrix;
};

} // namespace python
} // namespace linal

#endif // EIGEN_WRAPPER_H_
