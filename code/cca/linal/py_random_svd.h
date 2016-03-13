#ifndef PY_RANDOM_SVD_H_
#define PY_RANDOM_SVD_H_

// Imports from external projects
#include <Eigen/Dense>

// Imports from this project
#include "py_eigen_matrix.h"
#include "random_svd.h"

namespace linal {
namespace python {

class PyRandomSvd: public PyEigenMatrixXd
{

 public:
  // Static const members
  static const random::RandomSvd rSvd;

  // Constructor
  PyRandomSvd(const std::vector<std::vector<double> > &initial);
  ~PyRandomSvd(){}

  // Methods
  void GetRandomSvd();
  void GetRandomSvd(const int k);
  void GetRandomSvd(const int k, const int q);

  // Data members
  std::vector<std::vector<double> > U;
  std::vector<double> s;
  std::vector<std::vector<double> > V;

 private:
  void fill_svd(const std::vector<Eigen::MatrixXd> &matrices);

};

} // namespace python
} // namespace linal

#endif // PY_RANDOM_SVD_H_
