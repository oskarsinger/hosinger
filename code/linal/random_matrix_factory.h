#ifndef RANDOM_MATRIX_H_
#define RANDOM_MATRIX_H_

// Imports form external projects
#include <Eigen/Dense>

namespace linal {
namespace random {

class RandomMatrixFactory
{
 public:
  RandomMatrixFactory();
  Eigen::MatrixXd GetRankKMatrix(const int m, const int n, const int k);
  Eigen::MatrixXd GetNormalMatrix(const int m, const int n);
};

} // namespace random
} // namespace linal

#endif // RANDOM_MATRIX_H_
