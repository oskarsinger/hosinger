#ifndef PY_RANDOM_SVD_H_
#define PY_RANDOM_SVD_H_

// Imports from external projects
#include <Eigen/Dense>

// Imports from this project
#include "py_eigen_matrix.h"
#include "random_svd.h"

namespace linal {
namespace python {

class PyRandomSvd
{

 public:
  // Static const members
  static const random::RandomSvd rSvd;

  // Constructor
  PyRandomSvd() {}
  ~PyRandomSvd() {}

  // Methods
  std::vector<PyEigenMatrixXd> GetRandomSvd(const PyEigenMatrixXd &A) const;
  std::vector<PyEigenMatrixXd> GetRandomSvd(const PyEigenMatrixXd &A, const int k) const;
  std::vector<PyEigenMatrixXd> GetRandomSvd(const PyEigenMatrixXd &A, const int k, const int q) const;
  std::vector<PyEigenMatrixXd> WrapVector(const std::vector<Eigen::MatrixXd> &matrices) const;
};

} // namespace python
} // namespace linal

#endif // PY_RANDOM_SVD_H_
