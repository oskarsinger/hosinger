#ifndef RANDOM_ORTHONORMAL_BASIS_H_
#define RANDOM_ORTHONORMAL_BASIS_H_

// Imports from external projects
#include <Eigen/Dense>

namespace linal {
namespace random {

class RandomOrthonormalBasis
{
 public:
  RandomOrthonormalBasis();
  Eigen::MatrixXd get_epsilon_basis(Eigen::MatrixXd A, const double epsilon);
  Eigen::MatrixXd get_full_rank_basis(Eigen::MatrixXd A);
  Eigen::MatrixXd get_rank_k_basis(Eigen::MatrixXd A, const int k);
  Eigen::MatrixXd get_rank_k_basis(Eigen::MatrixXd A, const int k, const int q);
}

} // namespace random
} // namespace linal

#endif // RANDOM_ORTHONORMAL_BASIS_H_
