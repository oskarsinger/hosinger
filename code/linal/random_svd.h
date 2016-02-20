#ifndef RANDOM_SVD_H_
#define RANDOM_SVD_H_

// Imports from external projects
#include <Eigen/Dense>

namespace linal {
namespace random {

class RandomSVD
{

 public:
  RandomSVD();
  std::vector<MatrixXd> get_random_svd(MatrixXd A);
  std::vector<MatrixXd> get_random_svd(MatrixXd A, const int k);
  std::vector<MatrixXd> get_random_svd(MatrixXd A, const int k, const int q);
}

} // namespace random
} // namespace linal

#endif // RANDOM_SVD_H_
